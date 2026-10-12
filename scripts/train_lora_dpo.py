import argparse, dataclasses, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

def to_conversational(row):
    """Rows become TRL conversational preference records (role/content message lists)."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--val_size", type=int, default=20)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.data_path, encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[: args.val_size], rows[args.val_size:]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])
    print(f"train={len(train_ds)} val={len(val_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        bias="none", task_type="CAUSAL_LM",
    )

    cfg = dict(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta=args.beta,
        max_length=args.max_length,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    valid = {f.name for f in dataclasses.fields(DPOConfig)}
    ignored = sorted(set(cfg) - valid)
    if ignored:
        print("WARNING: DPOConfig ignores/lacks:", ignored)
    train_args = DPOConfig(**{k: v for k, v in cfg.items() if k in valid})

    try:
        trainer = DPOTrainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, peft_config=lora)
    except TypeError:
        trainer = DPOTrainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok, peft_config=lora)

    trainer.model.print_trainable_parameters()
    before = trainer.evaluate()
    print("EVAL BEFORE:", before)
    trainer.train()
    after = trainer.evaluate()
    print("EVAL AFTER:", after)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"config": vars(args), "eval_before": before, "eval_after": after,
                   "log_history": trainer.state.log_history}, f, indent=1)
    print("saved adapter to", adapter_dir)

if __name__ == "__main__":
    main()