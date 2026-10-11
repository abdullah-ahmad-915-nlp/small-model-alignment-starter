import argparse, json, math, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import format_prompt
from src.sft import build_sft_batch, response_logprobs_from_batch, IGNORE_INDEX

def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

def make_collate(tokenizer, max_length):
    def collate(rows):
        prompts = [format_prompt(r["prompt"]) for r in rows]
        responses = [r["response"] for r in rows]
        return build_sft_batch(tokenizer, prompts, responses, max_length)
    return collate

def loss_parts(model, batch):
    with torch.autocast("cuda", dtype=torch.float16):
        logps = response_logprobs_from_batch(model, batch)
    n_tokens = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum()
    return -logps.sum(), n_tokens

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        nll, n = loss_parts(model, batch)
        total += nll.item()
        count += n.item()
    model.train()
    return total / max(count, 1)

def save(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train_path", default="data/processed/sft_train.jsonl")
    p.add_argument("--val_path", default="data/processed/sft_val.jsonl")
    p.add_argument("--out_dir", default="outputs/sft")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_val_examples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "This script needs a CUDA GPU (run it on Kaggle)."
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.train_path, args.max_train_examples)
    val_rows = read_jsonl(args.val_path, args.max_val_examples)
    collate = make_collate(tokenizer, args.max_length)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=gen)
    val_loader = DataLoader(val_rows, batch_size=4, shuffle=False, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to("cuda:0")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    decay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.ndim >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.requires_grad and p_.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    print(f"train={len(train_rows)} val={len(val_rows)} micro-batches/epoch={len(train_loader)} optimizer steps={total_steps} warmup={warmup}")

    metrics_path = out_dir / "metrics.jsonl"
    best_val = float("inf")
    step, micro, window = 0, 0, []
    t0 = time.time()
    pbar = tqdm(total=total_steps, desc="SFT")
    opt.zero_grad(set_to_none=True)

    done = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            nll, n_tokens = loss_parts(model, batch)
            loss = nll / n_tokens.clamp_min(1)
            scaler.scale(loss / args.grad_accum).backward()
            window.append(loss.item())
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)

            rec = {"step": step, "train_loss": sum(window) / len(window),
                   "lr": sched.get_last_lr()[0], "grad_norm": grad_norm,
                   "elapsed_min": (time.time() - t0) / 60}
            window = []

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(model, val_loader)
                rec["val_loss"] = val
                if val < best_val:
                    best_val = val
                    save(model, tokenizer, out_dir / "best")
                    rec["saved_best"] = True
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if step % 10 == 0 or "val_loss" in rec:
                print(rec)

            if step >= total_steps:
                done = True
                break
        if done:
            break

    save(model, tokenizer, out_dir / "final")
    print(f"done. best val loss={best_val:.4f}  peak GPU mem="
          f"{torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB  "
          f"time={(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()