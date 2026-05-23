"""
Grouped Query Attention (GQA) — from scratch.

GQA generalizes both MHA and MQA:
  - MHA: n_kv_heads == n_q_heads   (each query head has its own K, V)
  - MQA: n_kv_heads == 1           (all query heads share one K, V)
  - GQA: 1 < n_kv_heads < n_q_heads  (groups of query heads share K, V)

Used in: LLaMA 2 (70B), Mistral 7B, Gemma, Falcon

Reference: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
           Ainslie et al., 2023 — https://arxiv.org/abs/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GQAConfig:
    n_embd:     int   = 256
    n_q_heads:  int   = 8
    n_kv_heads: int   = 2    # must divide n_q_heads evenly
    block_size: int   = 512
    dropout:    float = 0.1


class GroupedQueryAttention(nn.Module):
    """
    GQA: n_q_heads queries, n_kv_heads keys and values.
    Each group of (n_q_heads // n_kv_heads) query heads shares one K and one V head.

    This is the sweet spot between MHA quality and MQA efficiency.
    LLaMA 2 70B uses n_q_heads=64, n_kv_heads=8 → 8 query heads per KV group.
    """

    def __init__(self, config: GQAConfig):
        super().__init__()
        assert config.n_q_heads % config.n_kv_heads == 0, \
            "n_q_heads must be divisible by n_kv_heads"

        self.n_q_heads  = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_q_heads // config.n_kv_heads
        self.head_dim   = config.n_embd // config.n_q_heads
        self.n_embd     = config.n_embd

        self.W_q = nn.Linear(config.n_embd, config.n_q_heads * self.head_dim,  bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Q: (B, n_q_heads, T, head_dim)
        q = self.W_q(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, n_kv_heads, T, head_dim)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V to match n_q_heads by repeating each KV head n_groups times
        # k: (B, n_kv_heads, T, head_dim) → (B, n_q_heads, T, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(out))


def sweep_kv_heads(n_embd: int = 256, n_q_heads: int = 8, seq_len: int = 2048, n_layers: int = 32):
    """Compare KV cache sizes for different n_kv_heads values."""
    print(f"{'n_kv_heads':>12} | {'type':>6} | {'KV cache (MB)':>14}")
    print("-" * 40)
    for n_kv in [1, 2, 4, 8]:
        if n_q_heads % n_kv != 0:
            continue
        label = "MQA" if n_kv == 1 else ("MHA" if n_kv == n_q_heads else "GQA")
        head_dim = n_embd // n_q_heads
        mb = n_kv * head_dim * seq_len * n_layers * 2 * 2 / 1e6
        print(f"{n_kv:>12} | {label:>6} | {mb:>14.2f}")


if __name__ == "__main__":
    config = GQAConfig(n_embd=256, n_q_heads=8, n_kv_heads=2)
    attn = GroupedQueryAttention(config)
    x = torch.randn(2, 64, 256)
    out = attn(x)
    print(f"GQA output shape: {out.shape}")
    print(f"Q heads: {config.n_q_heads} | KV heads: {config.n_kv_heads} | Groups: {config.n_q_heads // config.n_kv_heads}")

    print("\n── KV cache sweep (n_embd=512, n_q=8, 2048 tokens, 32 layers) ──")
    sweep_kv_heads(512, 8, 2048, 32)
