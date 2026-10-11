import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
from src import data

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["sft", "hh", "uf"], required=True)
p.add_argument("--outdir", default="data/processed")
p.add_argument("--n_train", type=int, default=None)
p.add_argument("--n_val", type=int, default=1000)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if args.dataset == "sft":
    train, val = data.load_sft(n_train=args.n_train or 30000, n_val=args.n_val)
elif args.dataset == "hh":
    train, val = data.load_hh(tok, n_train=args.n_train or 15000, n_val=args.n_val)
else:
    train, val = data.load_ultrafeedback(tok, n_train=args.n_train or 15000, n_val=args.n_val)

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
for name, rows in (("train", train), ("val", val)):
    with open(out / f"{args.dataset}_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{args.dataset}: train={len(train)} val={len(val)}")
print("example:", json.dumps(train[0], ensure_ascii=False)[:600])