"""
SwiGLU + RMSNorm — from scratch.

Two architecture improvements that almost every modern LLM uses:

1. SwiGLU (Swish-Gated Linear Unit):
   Replaces FFN(x) = GELU(W1 x) W2
   With     FFN(x) = (SiLU(W1 x) ⊙ W3 x) W2
   — adds a gate that controls information flow, empirically better

2. RMSNorm (Root Mean Square Layer Normalization):
   Replaces LayerNorm: (x - mean) / (std + ε) * γ + β
   With     RMSNorm:   x / RMS(x) * γ
   — no bias, no mean subtraction, ~7% faster, equally effective

Used in: LLaMA, LLaMA 2, LLaMA 3, PaLM, Gemma, Mistral, Qwen

References:
  "GLU Variants Improve Transformer" — Noam Shazeer, 2020 (arxiv.org/abs/2002.05202)
  "Root Mean Square Layer Normalization" — Zhang & Sennrich, 2019 (arxiv.org/abs/1910.07467)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class RMSNorm(nn.Module):
    """
    RMSNorm: normalize by RMS instead of mean+std.

    RMS(x) = sqrt(1/d * sum(x_i^2))
    output  = x / RMS(x) * γ

    Benefits vs LayerNorm:
      - No centering (mean subtraction) — saves 2 passes over data
      - No bias parameter β
      - Empirically equal performance to LayerNorm
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # learned scale; no bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


class SwiGLU(nn.Module):
    """
    SwiGLU FFN: two parallel projections, one gated by SiLU.

    output = (SiLU(W1 x) ⊙ W3 x) W2

    The "gate" W3 x acts as a learned filter: positions where W3 x is small
    suppress the output from W1 x, adding a form of learned sparsity.

    Note on dimension:
      Standard FFN has inner dim = 4 * d_model.
      SwiGLU needs 2 × inner_dim parameters (W1 and W3), so to keep
      total params equal, use inner_dim = 2/3 * 4 * d_model ≈ 2.67 * d_model.
      LLaMA rounds this to the nearest multiple of 256.
    """

    def __init__(self, d_model: int, ffn_mult: float = 8/3, dropout: float = 0.1):
        super().__init__()
        inner = int(d_model * ffn_mult)
        # Round to multiple of 256 (hardware alignment)
        inner = ((inner + 255) // 256) * 256

        self.W1 = nn.Linear(d_model, inner, bias=False)  # "up" projection
        self.W3 = nn.Linear(d_model, inner, bias=False)  # gate projection
        self.W2 = nn.Linear(inner, d_model, bias=False)  # "down" projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W1(x))   # Swish activation (SiLU = x * sigmoid(x))
        x    = gate * self.W3(x)    # element-wise gate
        return self.drop(self.W2(x))


class StandardFFN(nn.Module):
    """Standard GELU FFN for comparison."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LLaMABlock(nn.Module):
    """
    Transformer block using RMSNorm + SwiGLU + RoPE (placeholder for RoPE here).
    This is essentially a LLaMA-style block.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1  = RMSNorm(n_embd)
        self.ln2  = RMSNorm(n_embd)
        self.ffn  = SwiGLU(n_embd, dropout=dropout)

        # Minimal attention (use RoPEAttention from project 10 in practice)
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) * self.head_dim**-0.5
                        + self.mask[:, :, :T, :T].log(), dim=-1)
        return self.W_o((att @ v).transpose(1, 2).contiguous().view(B, T, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def benchmark_norms(d: int = 512, T: int = 256, B: int = 8, n_iters: int = 1000):
    """Compare RMSNorm vs LayerNorm speed."""
    x = torch.randn(B, T, d)
    ln = nn.LayerNorm(d)
    rn = RMSNorm(d)

    for _ in range(50):  # warmup
        ln(x); rn(x)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        ln(x)
    ln_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        rn(x)
    rn_time = time.perf_counter() - t0

    print(f"LayerNorm: {ln_time*1000:.1f} ms  ({n_iters} iters)")
    print(f"RMSNorm:   {rn_time*1000:.1f} ms  ({n_iters} iters)")
    print(f"RMSNorm speedup: {ln_time / rn_time:.2f}×")


if __name__ == "__main__":
    x = torch.randn(2, 64, 256)
    rn  = RMSNorm(256)
    ffn = SwiGLU(256)
    blk = LLaMABlock(256, 8)

    print(f"RMSNorm output:  {rn(x).shape}")
    print(f"SwiGLU output:   {ffn(x).shape}")
    print(f"LLaMABlock out:  {blk(x).shape}")

    print("\n── Norm benchmark ──")
    benchmark_norms()
