import json, random
from pathlib import Path
from huggingface_hub import hf_hub_download

SEED, N = 42, 300
OUT = Path("eval_sets/alpaca_eval_300_seed42.json")

path = hf_hub_download(
    repo_id="tatsu-lab/alpaca_eval",
    filename="alpaca_eval.json",
    repo_type="dataset",
)
data = json.load(open(path))
print("total prompts:", len(data))

rng = random.Random(SEED)
idx = sorted(rng.sample(range(len(data)), N))

subset = [
    {"id": f"alpaca_eval_{i}", "instruction": data[i]["instruction"], "dataset": "alpaca_eval"}
    for i in idx
]

OUT.parent.mkdir(exist_ok=True)
json.dump(subset, open(OUT, "w"), indent=1)
print("saved", len(subset), "prompts to", OUT)