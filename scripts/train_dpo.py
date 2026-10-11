import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import format_prompt
from src.sft import response_logprobs_from_batch
from src.dpo import build_dpo_batch, compute_dpo_loss_from_logps, reward_metrics

PATHS = {"hh": ("data/processed/hh_train.jsonl", "data/processed/hh_val.jsonl"),
         "uf": ("data/processed/uf_train.jsonl", "data/processed/uf_val.jsonl")}

def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

def make_batch(tokenizer, rows, max_length):
    return build_dpo_batch(
        tokenizer,
        [format_prompt(r["prompt"]) for r in rows],
        [r["chosen"] for r in rows],
        [r["rejected"] for r in rows],
        max_length,
    )

def logps(model, batch, autocast):
    if autocast:
        with torch.autocast("cuda", dtype=torch.float16):
            c = response_logprobs_from_batch(model, batch["chosen"])
            r = response_logprobs_from_batch(model, batch["rejected"])
    else:
        c = response_logprobs_from_batch(model, batch["chosen"])
        r = response_logprobs_from_batch(model, batch["rejected"])
    return c, r

@torch.no_grad()
def cache_reference(ref, tokenizer, rows, max_length, bs):
    rcs, rrs = [], []
    for i in range(0, len(rows), bs):
        c, r = logps(ref, make_batch(tokenizer, rows[i:i + bs], max_length), autocast=False)
        rcs.append(c.cpu()); rrs.append(r.cpu())
    return torch.cat(rcs), torch.cat(rrs)

@torch.no_grad()
def evaluate(policy, tokenizer, rows, val_rc, val_rr, args):
    policy.eval()
    pcs, prs = [], []
    for i in range(0, len(rows), args.eval_batch):
        c, r = logps(policy, make_batch(tokenizer, rows[i:i + args.eval_batch], args.max_length), autocast=True)
        pcs.append(c.cpu()); prs.append(r.cpu())
    policy.train()
    pc, pr = torch.cat(pcs), torch.cat(prs)
    out = {"val_loss": compute_dpo_loss_from_logps(pc, pr, val_rc, val_rr, args.beta).item()}
    out.update({"val_" + k: v for k, v in reward_metrics(pc, pr, val_rc, val_rr, args.beta).items()})
    return out

def save(model, tokenizer, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True)
    p.add_argument("--dataset", choices=["hh", "uf"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_val_examples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--grad_ckpt", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Needs CUDA (run on Kaggle)."
    random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(args.sft_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_path, val_path = PATHS[args.dataset]
    train_rows = read_jsonl(train_path, args.max_train_examples)
    val_rows = read_jsonl(val_path, args.max_val_examples)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows, generator=gen)

    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, dtype=torch.float32).to("cuda:0")
    policy.config.use_cache = False
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable()
    policy.train()
    ref_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    ref = AutoModelForCausalLM.from_pretrained(args.sft_path, dtype=torch.float16).to(ref_dev)
    ref.eval(); ref.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    total_steps = len(loader) * args.epochs // args.grad_accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"dataset={args.dataset} train={len(train_rows)} val={len(val_rows)} steps={total_steps} "
          f"beta={args.beta} lr={args.lr} ref_device={ref_dev}")

    val_rc, val_rr = cache_reference(ref, tok, val_rows, args.max_length, args.eval_batch)

    metrics_path = out_dir / "metrics.jsonl"
    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)

    start = evaluate(policy, tok, val_rows, val_rc, val_rr, args)
    log({"step": 0, **start})

    best = (-1.0, -1e9)
    step, micro = 0, 0
    win = {"loss": [], "acc": [], "margin": []}
    t0 = time.time()
    pbar = tqdm(total=total_steps, desc="DPO")
    opt.zero_grad(set_to_none=True)

    done = False
    for epoch in range(args.epochs):
        for rows in loader:
            batch = make_batch(tok, rows, args.max_length)
            pc, pr = logps(policy, batch, autocast=True)
            with torch.no_grad():
                rc, rr = logps(ref, batch, autocast=False)
            rc, rr = rc.to(pc.device), rr.to(pc.device)

            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
            scaler.scale(loss / args.grad_accum).backward()

            m = reward_metrics(pc.detach(), pr.detach(), rc, rr, args.beta)
            win["loss"].append(loss.item()); win["acc"].append(m["reward_accuracy"]); win["margin"].append(m["reward_margin"])
            micro += 1
            if micro % args.grad_accum != 0:
                continue

            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip).item()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1; pbar.update(1)

            rec = {"step": step, "train_loss": sum(win["loss"]) / len(win["loss"]),
                   "train_reward_accuracy": sum(win["acc"]) / len(win["acc"]),
                   "train_reward_margin": sum(win["margin"]) / len(win["margin"]),
                   "grad_norm": gn, "elapsed_min": (time.time() - t0) / 60}
            win = {"loss": [], "acc": [], "margin": []}

            if step % args.eval_every == 0 or step == total_steps:
                rec.update(evaluate(policy, tok, val_rows, val_rc, val_rr, args))
                key = (rec["val_reward_accuracy"], rec["val_reward_margin"])
                if key > best:
                    best = key
                    save(policy, tok, out_dir / "best")
                    rec["saved_best"] = True
                log(rec)
            elif step % 10 == 0:
                log(rec)
            else:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            if step >= total_steps:
                done = True
                break
        if done:
            break

    save(policy, tok, out_dir / "final")
    print(f"done. best val (acc, margin)={best}  peak GPU0 mem="
          f"{torch.cuda.max_memory_allocated(0) / 1024**3:.1f} GiB  time={(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()