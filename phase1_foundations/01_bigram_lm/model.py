"""
Bigram Language Model

The simplest possible language model. Predicts the next token using only
the current token — no context, no attention, just a learned lookup table
of transition probabilities.

This is your baseline. Every improvement in the subsequent projects should
be measurable against the loss you get here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Each row i is the logit distribution over next tokens given current token i
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (B, T)  targets: (B, T)
        logits = self.token_embedding(idx)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_2d = logits.view(B * T, C)
            targets_1d = targets.view(B * T)
            loss = F.cross_entropy(logits_2d, targets_1d)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]          # last time step: (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)             # (B, T+1)
        return idx
