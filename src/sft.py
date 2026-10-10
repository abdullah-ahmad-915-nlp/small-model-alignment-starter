import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

def _pad_id(tokenizer):
    pid = getattr(tokenizer, "pad_token_id", None)
    if pid is None:
        pid = getattr(tokenizer, "eos_token_id", None)
    return 0 if pid is None else pid

def build_sft_batch(tokenizer, prompts, responses, max_length):
    assert len(prompts) == len(responses)
    eos = getattr(tokenizer, "eos_token_id", None)

    all_ids, all_labels = [], []
    for prompt, response in zip(prompts, responses):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response, add_special_tokens=False)
        if eos is not None:
            r_ids = r_ids + [eos]
        ids = (p_ids + r_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(p_ids) + r_ids)[:max_length]
        all_ids.append(ids)
        all_labels.append(labels)

    n = len(all_ids)
    width = max(len(x) for x in all_ids)
    pad = _pad_id(tokenizer)
    input_ids = torch.full((n, width), pad, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(all_ids, all_labels)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def response_logprobs_from_batch(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)

    token_logps = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(targets.shape)
    return (token_logps * mask).sum(dim=-1)

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return response_logprobs_from_batch(model, batch)

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps = response_logprobs_from_batch(model, batch)
    n_tokens = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum().clamp_min(1)
    return -logps.sum() / n_tokens