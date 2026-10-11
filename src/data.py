import random
from datasets import load_dataset

from src.prompts import format_prompt

SEED = 42

def _select(examples, n, seed, tokenizer=None, max_prompt_tokens=None):
    rng = random.Random(seed)
    examples = list(examples)
    rng.shuffle(examples)
    out = []
    for ex in examples:
        if tokenizer is not None and max_prompt_tokens is not None:
            n_tok = len(tokenizer.encode(format_prompt(ex["prompt"]), add_special_tokens=False))
            if n_tok > max_prompt_tokens:
                continue
        out.append(ex)
        if len(out) == n:
            break
    if len(out) < n:
        print(f"WARNING: wanted {n} examples, only {len(out)} available after filtering")
    return out

def _clean_ultrachat(split):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    out = []
    for row in ds:
        m = row["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            prompt, response = m[0]["content"].strip(), m[1]["content"].strip()
            if prompt and response:
                out.append({"prompt": prompt, "response": response})
    return out

def load_sft(n_train=30000, n_val=1000, seed=SEED):
    train = _select(_clean_ultrachat("train_sft"), n_train, seed)
    val = _select(_clean_ultrachat("test_sft"), n_val, seed)
    return train, val

def _parse_hh(text):
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    human_part, assistant_part = text.split("\n\nAssistant:")
    prompt = human_part.split("\n\nHuman:")[1].strip()
    response = assistant_part.strip()
    if not prompt or not response:
        return None
    return prompt, response

def _clean_hh(split):
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    out = []
    for row in ds:
        c, r = _parse_hh(row["chosen"]), _parse_hh(row["rejected"])
        if c is None or r is None or c[0] != r[0] or c[1] == r[1]:
            continue
        out.append({"prompt": c[0], "chosen": c[1], "rejected": r[1]})
    return out

def load_hh(tokenizer=None, n_train=15000, n_val=1000, seed=SEED, max_prompt_tokens=256):
    train = _select(_clean_hh("train"), n_train, seed, tokenizer, max_prompt_tokens)
    val = _select(_clean_hh("test"), n_val, seed, tokenizer, max_prompt_tokens)
    return train, val

def _clean_uf(split):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    out = []
    for row in ds:
        c, r = row["chosen"], row["rejected"]
        if len(c) != 2 or len(r) != 2:
            continue
        prompt = c[0]["content"].strip()
        chosen, rejected = c[1]["content"].strip(), r[1]["content"].strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        if r[0]["content"].strip() != prompt:
            continue
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out

def load_ultrafeedback(tokenizer=None, n_train=15000, n_val=1000, seed=SEED, max_prompt_tokens=256):
    train = _select(_clean_uf("train_prefs"), n_train, seed, tokenizer, max_prompt_tokens)
    val = _select(_clean_uf("test_prefs"), n_val, seed, tokenizer, max_prompt_tokens)
    return train, val