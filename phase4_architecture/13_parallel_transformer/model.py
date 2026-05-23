"""
Parallel Transformer Block (PaLM-style) — from scratch.

Standard transformer: sequential attention → FFN
  x = x + Attn(LN(x))
  x = x + FFN(LN(x))

Parallel transformer: attention AND FFN computed simultaneously on the same input
  x = x + Attn(LN(x)) + FFN(LN(x))

Benefits:
  1. ~15% speedup on TPU/GPU due to better parallelism
  2. Fewer sequential operations → shorter critical path
  3. Slightly lower quality on small models, roughly equal on large models

Reference: "PaLM: Scaling Language Modeling with Pathways"
           Chowdhery et al., 2022 — https://arxiv.org/abs/2204.02311
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ParallelConfig:
    vocab_size:  int   = 65
    block_size:  int   = 256
    n_embd:      int   = 256
    n_head:      int   = 8
    n_layer:     int   = 6
    dropout:     float = 0.1


class ParallelTransformerBlock(nn.Module):
    """
    Parallel attention + FFN block.

    The key insight: both Attn and FFN read from the SAME LN(x).
    This removes the sequential dependency, enabling their computations
    to overlap on hardware with separate attention and GEMM units.

    Memory note: we share the layer norm between attention and FFN.
    """

    def __init__(self, config: ParallelConfig):
        super().__init__()
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd   = config.n_embd

        self.ln = nn.LayerNorm(config.n_embd)  # SINGLE shared LN for both sub-layers

        # Attention projections
        self.W_q   = nn.Linear(config.n_embd, config.n_embd,     bias=False)
        self.W_k   = nn.Linear(config.n_embd, config.n_embd,     bias=False)
        self.W_v   = nn.Linear(config.n_embd, config.n_embd,     bias=False)
        self.W_attn = nn.Linear(config.n_embd, config.n_embd,    bias=False)

        # FFN projections
        self.W_up   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.W_down = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        self.drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_attn(out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.gelu(self.W_up(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        # Parallel: both read from the same normalized input
        return self.drop(x + self._attention(h) + self._ffn(h))


class SequentialTransformerBlock(nn.Module):
    """Standard sequential block for comparison."""

    def __init__(self, config: ParallelConfig):
        super().__init__()
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd   = config.n_embd

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_up   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.W_down = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.ln1(x)
        B, T, C = h1.shape
        q = self.W_q(h1).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(h1).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(h1).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1) @ v
        x = x + self.W_attn(att.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.W_down(F.gelu(self.W_up(self.ln2(x))))
        return self.drop(x)


if __name__ == "__main__":
    import time
    config = ParallelConfig()
    x = torch.randn(4, 128, 256)

    parallel = ParallelTransformerBlock(config)
    sequential = SequentialTransformerBlock(config)

    print(f"Parallel params:   {sum(p.numel() for p in parallel.parameters()):,}")
    print(f"Sequential params: {sum(p.numel() for p in sequential.parameters()):,}")

    out_p = parallel(x)
    out_s = sequential(x)
    print(f"Parallel output:   {out_p.shape}")
    print(f"Sequential output: {out_s.shape}")
