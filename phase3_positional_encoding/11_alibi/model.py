"""
ALiBi (Attention with Linear Biases) — from scratch.

Instead of adding positional information to embeddings, ALiBi adds a fixed
linear BIAS to attention scores AFTER the dot product:

  score(q_i, k_j) = q_i · k_j / sqrt(d) - m_h * (i - j)

where m_h is a head-specific slope and (i - j) is the relative distance.

Key properties:
  - No positional embeddings at all (zero position parameters)
  - Extrapolates to longer sequences than seen during training
  - Different heads use different slopes → multi-scale relative bias
  - Penalizes attending to distant tokens (recency bias baked in)

Used in: BLOOM (176B), MPT, OPT (partially)

Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
           Press et al., 2021 — https://arxiv.org/abs/2108.12409
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    The original paper uses slopes that form a geometric sequence:
      If n_heads is a power of 2: m_h = 2^(-8h/n_heads) for h in 1..n_heads
      Otherwise: interpolate between the nearest power-of-2 schedule
    """
    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _slopes_power_of_2(n_heads)
    else:
        # Nearest power of 2 below and above, then interpolate
        n_lower = 2 ** math.floor(math.log2(n_heads))
        slopes_lower = _slopes_power_of_2(n_lower)
        slopes_upper = _slopes_power_of_2(2 * n_lower)
        # Take every other slope from the upper sequence to fill in
        slopes = slopes_lower + slopes_upper[0::2][: n_heads - n_lower]

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(n_heads: int, seq_len: int, device=None) -> torch.Tensor:
    """
    Build the ALiBi bias tensor: (1, n_heads, seq_len, seq_len)

    bias[h, i, j] = -slope_h * (i - j)  for j <= i (causal)
                    -inf                  for j > i  (future masking)
    """
    slopes = get_alibi_slopes(n_heads).to(device)  # (n_heads,)

    # Relative positions: (1, seq_len) - (seq_len, 1) → (seq_len, seq_len)
    positions = torch.arange(seq_len, device=device)
    rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # j - i (negative = past)

    # ALiBi uses -|i - j| = rel_pos for j <= i (past positions are negative rel_pos)
    # bias = slopes * rel_pos: past tokens get negative bias proportional to distance
    bias = slopes.view(n_heads, 1, 1) * rel_pos.unsqueeze(0)  # (n_heads, T, T)

    # Apply causal mask: future positions → -inf
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    bias = bias.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))

    return bias.unsqueeze(0)  # (1, n_heads, T, T)


class ALiBiAttention(nn.Module):
    """
    Causal self-attention with ALiBi positional bias.

    No positional embeddings in the input — position is encoded purely
    via the bias added to attention scores.
    """

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.n_embd   = n_embd

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Precompute bias up to max_seq_len
        bias = build_alibi_bias(n_head, max_seq_len)
        self.register_buffer("alibi_bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale    # (B, n_head, T, T)

        # Add ALiBi bias (handles causal masking internally via -inf)
        scores = scores + self.alibi_bias[:, :, :T, :T]

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(out))


class ALiBiTransformer(nn.Module):
    """Full transformer using ALiBi (no positional embeddings in input)."""

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # NO pos_emb!
        self.blocks  = nn.ModuleList([
            nn.ModuleDict({
                "ln1":  nn.LayerNorm(n_embd),
                "attn": ALiBiAttention(n_embd, n_head, max_seq_len, dropout),
                "ln2":  nn.LayerNorm(n_embd),
                "ffn":  nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = x + block["attn"](block["ln1"](x))
            x = x + block["ffn"](block["ln2"](x))
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


if __name__ == "__main__":
    slopes = get_alibi_slopes(8)
    print(f"ALiBi slopes (8 heads): {slopes.tolist()}")

    attn = ALiBiAttention(n_embd=256, n_head=8, max_seq_len=512)
    x = torch.randn(2, 64, 256)
    out = attn(x)
    print(f"Output shape: {out.shape}")
