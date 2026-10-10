import torch
import torch.nn.functional as F

from src.sft import build_sft_batch, response_logprobs_from_batch

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }

def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    pc = policy_chosen_logps.float()
    pr = policy_rejected_logps.float()
    rc = reference_chosen_logps.detach().float()
    rr = reference_rejected_logps.detach().float()
    logits = beta * ((pc - pr) - (rc - rr))
    return -F.logsigmoid(logits).mean()

def implicit_rewards(pc, pr, rc, rr, beta):
    r_w = beta * (pc - rc)
    r_l = beta * (pr - rr)
    return r_w, r_l

def reward_metrics(pc, pr, rc, rr, beta):
    r_w, r_l = implicit_rewards(pc, pr, rc, rr, beta)
    return {
        "reward_accuracy": (r_w > r_l).float().mean().item(),
        "reward_margin": (r_w - r_l).mean().item(),
    }

def compute_dpo_loss(policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    pc = response_logprobs_from_batch(policy_model, batch["chosen"])
    pr = response_logprobs_from_batch(policy_model, batch["rejected"])

    with torch.no_grad():
        rc = response_logprobs_from_batch(reference_model, batch["chosen"])
        rr = response_logprobs_from_batch(reference_model, batch["rejected"])

    device = pc.device
    return compute_dpo_loss_from_logps(pc, pr, rc.to(device), rr.to(device), beta)