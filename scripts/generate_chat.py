import argparse, json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, RepetitionPenaltyLogitsProcessor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--eval_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--dataset", default="persona_style_eval")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model = model.to(device).eval()

    items = json.load(open(args.eval_path, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"]: r for r in json.load(open(out_path))} if out_path.exists() else {}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate")

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i: i + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": b["instruction"]}], tokenize=False, add_generation_prompt=True) for b in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        procs = None
        if args.repetition_penalty != 1.0:
            procs = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(
                penalty=float(args.repetition_penalty), prompt_ignore_length=enc["input_ids"].shape[1])])
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, repetition_penalty=1.0, logits_processor=procs)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for b, t in zip(batch, texts):
            done[b["id"]] = {"id": b["id"], "instruction": b["instruction"], "output": t.strip(), "generator": args.generator, "dataset": args.dataset}
        json.dump([done[it["id"]] for it in items if it["id"] in done], open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"saved {len(done)}/{len(items)}")

if __name__ == "__main__":
    main()