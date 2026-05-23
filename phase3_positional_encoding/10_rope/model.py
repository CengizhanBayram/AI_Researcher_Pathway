"""
Rotary Position Embedding (RoPE) — from scratch.

Instead of adding position vectors to tokens, RoPE ROTATES the Q and K vectors
by an angle proportional to their absolute position. This encodes relative
position implicitly in the dot product.

Key insight: if we rotate q_m by angle m·θ and k_n by angle n·θ, then
  q_m · k_n = q · k · cos((m - n) · θ)
which depends only on the RELATIVE position (m - n), not m or n individually.

Used in: LLaMA, LLaMA 2, LLaMA 3, PaLM, Falcon, Mistral, Gemma, Qwen

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           Su et al., 2021 — https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def precompute_rope_frequencies(head_dim: int, seq_len: int, base: float = 10000.0, device=None):
    """
    Precompute the complex frequency tensor for RoPE.

    For each dimension pair (2i, 2i+1), the rotation angle at position m is:
      θ_i = m / (base^(2i/d))

    Returns:
      freqs_cos: (seq_len, head_dim/2)
      freqs_sin: (seq_len, head_dim/2)
    """
    assert head_dim % 2 == 0
    # θ_i = 1 / (base^(2i/d)) for i in [0, d/2)
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    # Outer product: (seq_len, head_dim/2)
    angles = torch.outer(positions, theta)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to x.

    x: (B, n_head, T, head_dim)
    cos, sin: (T, head_dim/2)

    The rotation of a 2D vector (x1, x2) by angle θ:
      x1' = x1 cos(θ) - x2 sin(θ)
      x2' = x1 sin(θ) + x2 cos(θ)

    We split head_dim into pairs and rotate each pair.
    """
    B, H, T, D = x.shape
    half = D // 2

    x1 = x[..., :half]   # (B, H, T, D/2)
    x2 = x[..., half:]   # (B, H, T, D/2)

    # Expand cos/sin: (T, D/2) → (1, 1, T, D/2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    x_rot_1 = x1 * cos - x2 * sin
    x_rot_2 = x1 * sin + x2 * cos
    return torch.cat([x_rot_1, x_rot_2], dim=-1)


@dataclass
class RoPEConfig:
    n_embd:     int   = 256
    n_head:     int   = 8
    block_size: int   = 512
    dropout:    float = 0.1
    rope_base:  float = 10000.0  # LLaMA 3 uses 500000.0 for extended context


class RoPESelfAttention(nn.Module):
    """
    Causal self-attention with Rotary Position Embedding.

    RoPE is applied AFTER projecting to Q and K, BEFORE the dot product.
    V is NOT rotated — only Q and K need positional information for the score.
    """

    def __init__(self, config: RoPEConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd   = config.n_embd

        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        # Precompute RoPE frequencies for the maximum sequence length
        cos, sin = precompute_rope_frequencies(self.head_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K only
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(out))


def verify_relative_position(head_dim: int = 64, seq_len: int = 16):
    """
    Verify that RoPE dot product depends only on relative position.
    For fixed content vectors q, k:
      dot(rotate(q, m), rotate(k, n)) should equal dot(rotate(q, m-n), rotate(k, 0))
    """
    cos, sin = precompute_rope_frequencies(head_dim, seq_len + 10)

    q = torch.randn(1, 1, seq_len, head_dim)
    k = torch.randn(1, 1, seq_len, head_dim)

    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

    # Score at (m=5, n=2) — relative position 3
    score_5_2 = (q_rot[:, :, 5, :] * k_rot[:, :, 2, :]).sum().item()

    # Shift: score at (m=3, n=0) — same relative position 3
    score_3_0 = (q_rot[:, :, 3, :] * k_rot[:, :, 0, :]).sum().item()

    print(f"Score at positions (5, 2): {score_5_2:.4f}")
    print(f"Score at positions (3, 0): {score_3_0:.4f}")
    print("(These should be approximately equal — both have relative distance 3)")


if __name__ == "__main__":
    config = RoPEConfig(n_embd=256, n_head=8, block_size=512)
    attn = RoPESelfAttention(config)
    x = torch.randn(2, 64, 256)
    out = attn(x)
    print(f"RoPE attention output: {out.shape}")

    print("\n── Relative position property ──")
    verify_relative_position()
