import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import format_prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--eval_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--dataset", default="alpaca_eval")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype).to(device)
    model.eval()

    items = json.load(open(args.eval_path))
    if args.limit:
        items = items[: args.limit]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        done = {r["id"]: r for r in json.load(open(out_path))}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate")

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        prompts = [format_prompt(b["instruction"]) for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for b, t in zip(batch, texts):
            done[b["id"]] = {
                "id": b["id"],
                "instruction": b["instruction"],
                "output": t.strip(),
                "generator": args.generator,
                "dataset": args.dataset,
            }
        ordered = [done[it["id"]] for it in items if it["id"] in done]
        json.dump(ordered, open(out_path, "w"), indent=1)
        print(f"saved {len(done)}/{len(items)}")

if __name__ == "__main__":
    main()