import argparse, json, random
from pathlib import Path

def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True)
    p.add_argument("--dpo_path", required=True)
    p.add_argument("--eval_path", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sft, dpo = load(args.sft_path), load(args.dpo_path)
    extra = load(args.eval_path) if args.eval_path else {}

    ids = [i for i in sft if i in dpo]
    skipped = [i for i in sft if i not in dpo] + [i for i in dpo if i not in sft]
    if skipped:
        print(f"WARNING: {len(skipped)} ids present in only one file were skipped")

    rng = random.Random(args.seed)
    flags = [True] * (len(ids) // 2) + [False] * (len(ids) - len(ids) // 2)
    rng.shuffle(flags)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_identical, n_empty = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for i, dpo_is_a in zip(ids, flags):
            s, d = sft[i]["output"], dpo[i]["output"]
            a_text, b_text = (d, s) if dpo_is_a else (s, d)
            row = {
                "id": i,
                "instruction": sft[i]["instruction"],
                "response_a": a_text,
                "response_b": b_text,
                "dataset": sft[i].get("dataset"),
            }
            for k in ("category", "expected_behavior"):
                if k in extra.get(i, {}):
                    row[k] = extra[i][k]
            identical = s.strip() == d.strip()
            n_identical += identical
            n_empty += (not s.strip()) + (not d.strip())
            row["hidden"] = {
                "a_model": "dpo" if dpo_is_a else "sft",
                "b_model": "sft" if dpo_is_a else "dpo",
                "identical": identical,
                "sft_generator": sft[i].get("generator"),
                "dpo_generator": dpo[i].get("generator"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(ids)} pairs to {out}")
    print(f"DPO shown as Response A in {sum(flags)} pairs, as Response B in {len(flags) - sum(flags)}")
    print(f"identical outputs: {n_identical} | empty outputs: {n_empty}")

if __name__ == "__main__":
    main()