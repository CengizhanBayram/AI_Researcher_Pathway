"""
Multi-Query Attention (MQA) — from scratch.

Key idea: all query heads share a SINGLE key head and a SINGLE value head.
This reduces KV cache size from O(n_head * head_dim) to O(head_dim) per token,
drastically cutting memory bandwidth during inference.

Reference: "Fast Transformer Decoding: One Write-Head is All You Need"
           Shazeer, 2019 — https://arxiv.org/abs/1911.02150
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MQAConfig:
    n_embd:    int   = 256
    n_q_heads: int   = 8    # number of query heads (normal)
    block_size: int  = 512
    dropout:   float = 0.1


class MultiQueryAttention(nn.Module):
    """
    MQA: n_q_heads query projections, but only 1 key and 1 value projection.

    Memory comparison at inference (per token, per layer):
      MHA:  n_head × head_dim × 2  (K and V)
      MQA:  head_dim × 2           (1 K, 1 V shared across all query heads)

    Speed comparison:
      MQA can be 1.5–2× faster during autoregressive decoding because
      KV cache fits in faster memory tiers.
    """

    def __init__(self, config: MQAConfig):
        super().__init__()
        assert config.n_embd % config.n_q_heads == 0
        self.n_q_heads = config.n_q_heads
        self.head_dim  = config.n_embd // config.n_q_heads
        self.n_embd    = config.n_embd

        # n_q_heads query projections
        self.W_q = nn.Linear(config.n_embd, config.n_embd,    bias=False)
        # Single shared K and V (head_dim, not n_embd)
        self.W_k = nn.Linear(config.n_embd, self.head_dim, bias=False)
        self.W_v = nn.Linear(config.n_embd, self.head_dim, bias=False)
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

        # Q: (B, T, n_embd) → (B, n_q_heads, T, head_dim)
        q = self.W_q(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, T, head_dim) → (B, 1, T, head_dim)  — single head
        k = self.W_k(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        # Broadcast k/v across all query heads
        # k: (B, 1, T, head_dim) → automatically broadcasts with q: (B, n_q_heads, T, head_dim)
        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale   # (B, n_q_heads, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # v broadcasts: (B, 1, T, head_dim) → (B, n_q_heads, T, head_dim)
        out = att @ v                                                 # (B, n_q_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(out))


class MQATransformerBlock(nn.Module):
    def __init__(self, config: MQAConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.attn = MultiQueryAttention(config)
        self.ffn  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def compare_kv_cache_size(n_heads: int, head_dim: int, seq_len: int, n_layers: int):
    """Show the memory saving from MQA vs MHA."""
    mha_bytes = n_heads * head_dim * seq_len * n_layers * 2 * 2  # 2 for K,V; 2 for fp16
    mqa_bytes = 1      * head_dim * seq_len * n_layers * 2 * 2
    print(f"MHA KV cache: {mha_bytes / 1e6:.2f} MB")
    print(f"MQA KV cache: {mqa_bytes / 1e6:.2f} MB")
    print(f"Reduction:    {mha_bytes / mqa_bytes:.1f}×")


if __name__ == "__main__":
    config = MQAConfig(n_embd=256, n_q_heads=8, block_size=512)
    attn = MultiQueryAttention(config)
    x = torch.randn(2, 64, 256)
    out = attn(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    print("\n── KV Cache comparison (8 heads, 32 head_dim, 2048 tokens, 32 layers) ──")
    compare_kv_cache_size(n_heads=8, head_dim=32, seq_len=2048, n_layers=32)
