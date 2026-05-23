"""
Speculative Decoding — from scratch.

Key idea: use a small, fast "draft" model to propose γ tokens in parallel,
then use the large "target" model to verify all of them in a single forward pass.

If the draft is usually right, you get γ tokens for the cost of 1 target pass.
Average accepted tokens per target step: γ' ≈ γ * α where α = acceptance rate.
Speedup ≈ γ' / (1 + cost(draft)) — can be 2-3× over standard generation.

Acceptance criterion (same output distribution as the target):
  For draft token x with draft prob q(x) and target prob p(x):
    - Accept with probability min(1, p(x) / q(x))
    - If rejected, sample a corrected token from max(0, p - q) / Z

Reference: "Speculative Decoding" — Chen et al., 2023 — https://arxiv.org/abs/2302.01318
           "Fast Inference from Transformers via Speculative Decoding" — Leviathan et al., 2023
"""

import torch
import torch.nn.functional as F
from typing import Optional


def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Speculative decoding: draft γ tokens, verify with target in 1 pass.

    Returns:
        output_ids: (1, original_len + new_tokens)
        stats: dict with acceptance rate and total target calls
    """
    device = input_ids.device
    n_target_calls = 0
    n_tokens_generated = 0
    n_accepted_total = 0

    while n_tokens_generated < max_new_tokens:
        remaining = max_new_tokens - n_tokens_generated
        gamma_step = min(gamma, remaining)

        # ── Step 1: Draft model generates γ candidate tokens ──────────────
        draft_tokens = []
        draft_probs  = []
        current_ids  = input_ids.clone()

        with torch.no_grad():
            for _ in range(gamma_step):
                logits = draft_model(current_ids)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, 1)

                draft_tokens.append(next_t)
                draft_probs.append(probs)
                current_ids = torch.cat([current_ids, next_t], dim=1)

        # ── Step 2: Target model scores all γ+1 positions in ONE pass ─────
        draft_sequence = current_ids  # prompt + γ draft tokens
        with torch.no_grad():
            target_logits = target_model(draft_sequence)
            n_target_calls += 1

        # Get target probs for positions corresponding to draft choices
        # target_logits[:, prompt_len-1 : prompt_len+γ-1] → p(x_i | x<i)
        prompt_len = input_ids.shape[1]
        target_probs_seq = F.softmax(target_logits[:, prompt_len - 1 : prompt_len + gamma_step - 1, :] / temperature, dim=-1)

        # ── Step 3: Accept/reject each draft token ────────────────────────
        n_accepted = 0
        for i in range(gamma_step):
            x_i   = draft_tokens[i]                         # (1, 1)
            q_x   = draft_probs[i].gather(-1, x_i)          # draft prob
            p_x   = target_probs_seq[:, i, :].gather(-1, x_i)  # target prob

            accept_prob = torch.clamp(p_x / (q_x + 1e-10), max=1.0)
            u = torch.rand_like(accept_prob)

            if u.item() <= accept_prob.item():
                input_ids = torch.cat([input_ids, x_i], dim=1)
                n_accepted += 1
                n_tokens_generated += 1
            else:
                # Reject: sample from corrected distribution max(0, p - q)
                corrected = F.relu(target_probs_seq[:, i, :] - draft_probs[i])
                z = corrected.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                corrected = corrected / z
                corrected_token = torch.multinomial(corrected, 1)
                input_ids = torch.cat([input_ids, corrected_token], dim=1)
                n_tokens_generated += 1
                break  # stop this draft batch; restart from corrected token

        n_accepted_total += n_accepted

        # If all γ tokens accepted, also use the target's prediction for position γ+1
        if n_accepted == gamma_step and n_tokens_generated < max_new_tokens:
            final_logits = target_logits[:, prompt_len + gamma_step - 1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(final_logits, min(top_k, final_logits.size(-1)))
                final_logits[final_logits < v[:, [-1]]] = float("-inf")
            bonus_token = torch.multinomial(F.softmax(final_logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, bonus_token], dim=1)
            n_tokens_generated += 1

    total_draft_tokens = n_accepted_total + n_target_calls  # approx
    acceptance_rate = n_accepted_total / max(1, n_tokens_generated)

    stats = {
        "n_target_calls":   n_target_calls,
        "n_tokens_generated": n_tokens_generated,
        "acceptance_rate":  acceptance_rate,
        "avg_tokens_per_target_call": n_tokens_generated / max(1, n_target_calls),
    }
    return input_ids, stats


def demo_speculative():
    """Demonstrate speculative decoding with tiny models."""
    import sys
    sys.path.insert(0, str(__file__).split("phase5")[0])

    # Build a small draft and a larger target
    from phase1_foundations.model_02 import GPT, GPTConfig

    draft_cfg = GPTConfig(vocab_size=65, block_size=128, n_embd=64,  n_head=2, n_layer=2)
    target_cfg = GPTConfig(vocab_size=65, block_size=128, n_embd=128, n_head=4, n_layer=4)

    draft  = GPT(draft_cfg).eval()
    target = GPT(target_cfg).eval()

    # Wrap to return full logits (not (logits, loss) tuple)
    def draft_fn(ids):  return draft(ids)[0]
    def target_fn(ids): return target(ids)[0]

    prompt = torch.zeros((1, 8), dtype=torch.long)
    out, stats = speculative_decode(draft_fn, target_fn, prompt, max_new_tokens=32, gamma=4)
    print(f"Generated {stats['n_tokens_generated']} tokens")
    print(f"Target calls: {stats['n_target_calls']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Avg tokens per target call: {stats['avg_tokens_per_target_call']:.2f}")
    print(f"(Baseline without speculative: {stats['n_tokens_generated']} target calls)")


if __name__ == "__main__":
    print("Speculative decoding module loaded.")
    print("Run demo_speculative() to test (requires Phase 1 GPT models).")
