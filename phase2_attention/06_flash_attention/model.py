"""
Flash Attention — tiled, IO-aware attention — from scratch (CPU/Python version).

Standard attention materializes the full (T×T) attention matrix in HBM (GPU DRAM).
Flash Attention avoids this by:
  1. Splitting Q, K, V into tiles that fit in SRAM (on-chip cache)
  2. Computing the softmax incrementally using the online softmax algorithm
  3. Never writing the full T×T matrix to HBM

This implementation is a pure Python/PyTorch reference that demonstrates the
tiling + online-softmax algorithm. The real speedup only appears in CUDA
(where HBM vs SRAM bandwidth matters). Use torch.nn.functional.scaled_dot_product_attention
in production — it calls FlashAttention kernels automatically.

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
           Dao et al., 2022 — https://arxiv.org/abs/2205.14135
"""

import math
import torch
import torch.nn as nn


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Standard O(T²) attention — materializes the full T×T matrix."""
    B, H, T, d = q.shape
    scale = d ** -0.5
    scores = q @ k.transpose(-2, -1) * scale

    if causal:
        mask = torch.tril(torch.ones(T, T, device=q.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 32,
    causal: bool = True,
) -> torch.Tensor:
    """
    Tiled attention with online (numerically stable) softmax.

    For each query tile Qi:
      - Loop over key/value tiles Kj, Vj
      - Maintain running max (m) and running sum (l) for online softmax
      - Accumulate output O incrementally
      - Never store the full T×T matrix

    Memory: O(T * block_size) instead of O(T²)

    Online softmax trick:
      When we see a new max m_new > m_old:
        l_new = exp(m_old - m_new) * l_old + sum(exp(s_j - m_new))
        O_new = exp(m_old - m_new) * O_old + exp(s_j - m_new) @ Vj
    """
    B, H, T, d = q.shape
    scale = d ** -0.5
    O = torch.zeros_like(q)

    for i in range(0, T, block_size):
        qi = q[:, :, i : i + block_size, :]   # (B, H, Bi, d)
        Bi = qi.shape[2]

        # Running accumulators for online softmax
        m_i = torch.full((B, H, Bi, 1), float("-inf"), device=q.device)  # running max
        l_i = torch.zeros((B, H, Bi, 1), device=q.device)               # running sum of exp
        o_i = torch.zeros((B, H, Bi, d), device=q.device)               # running output

        for j in range(0, T, block_size):
            # Causal masking: skip future key blocks entirely
            if causal and j > i + block_size - 1:
                break

            kj = k[:, :, j : j + block_size, :]
            vj = v[:, :, j : j + block_size, :]

            # Compute scores for this tile: (B, H, Bi, Bj)
            s_ij = qi @ kj.transpose(-2, -1) * scale

            if causal:
                # Apply causal mask within the tile
                qi_pos = torch.arange(i, i + Bi, device=q.device).unsqueeze(1)  # (Bi, 1)
                kj_pos = torch.arange(j, j + kj.shape[2], device=q.device).unsqueeze(0)  # (1, Bj)
                tile_mask = qi_pos >= kj_pos  # (Bi, Bj)
                s_ij = s_ij.masked_fill(~tile_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Online softmax update
            m_ij = s_ij.max(dim=-1, keepdim=True).values  # (B, H, Bi, 1)
            m_new = torch.maximum(m_i, m_ij)

            exp_s = torch.exp(s_ij - m_new)               # (B, H, Bi, Bj)
            l_new = torch.exp(m_i - m_new) * l_i + exp_s.sum(dim=-1, keepdim=True)

            o_i = torch.exp(m_i - m_new) * o_i + exp_s @ vj

            m_i = m_new
            l_i = l_new

        O[:, :, i : i + Bi, :] = o_i / l_i  # normalize

    return O


class FlashAttentionLayer(nn.Module):
    """Drop-in replacement for CausalSelfAttention using tiled flash attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int = 512, dropout: float = 0.1, tile_size: int = 32):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head

        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd,     bias=False)
        self.resid_drop = nn.Dropout(dropout)
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        out = flash_attention_tiled(q, k, v, block_size=self.tile_size, causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


def verify_equivalence(B: int = 2, H: int = 4, T: int = 64, d: int = 32):
    """Verify flash attention matches naive attention numerically."""
    torch.manual_seed(42)
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    out_naive = naive_attention(q, k, v, causal=True)
    out_flash = flash_attention_tiled(q, k, v, block_size=16, causal=True)

    max_diff = (out_naive - out_flash).abs().max().item()
    print(f"Max absolute difference (naive vs flash): {max_diff:.2e}")
    assert max_diff < 1e-5, f"Too large: {max_diff}"
    print("Outputs match!")


if __name__ == "__main__":
    verify_equivalence()
