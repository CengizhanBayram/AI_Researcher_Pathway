"""
Sliding Window Attention (SWA) — from scratch.

Instead of attending to all T positions (O(T²)), each token attends only
to a local window of w tokens: O(T·w).

Used in: Mistral 7B, Longformer, BigBird (as one of multiple patterns)

Reference: "Longformer: The Long-Document Transformer"
           Beltagy et al., 2020 — https://arxiv.org/abs/2004.05150
           "Mistral 7B" — Jiang et al., 2023 — https://arxiv.org/abs/2310.06825
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Causal sliding window attention.
    Each position i attends to positions max(0, i-w+1) ... i.

    The window_size w controls the tradeoff:
      - Small w: fast but can miss long-range dependencies
      - Large w → T: equivalent to full attention

    Mistral 7B uses w=4096 with T up to 32768 via rolling buffer KV cache.
    """

    def __init__(self, n_embd: int, n_head: int, window_size: int = 128, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head     = n_head
        self.head_dim   = n_embd // n_head
        self.window_size = window_size

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D, W = self.n_head, self.head_dim, self.window_size

        q = self.W_q(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = self.W_k(x).view(B, T, H, D).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, T, H, D).permute(0, 2, 1, 3)

        scale = D ** -0.5

        # Build the sliding window mask: (T, T) — True = attend, False = mask
        # Position i attends to j if i - W + 1 <= j <= i
        row_idx = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
        col_idx = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        causal_mask  = col_idx <= row_idx                         # lower triangular
        window_mask  = col_idx >= (row_idx - W + 1)              # within window
        attend_mask  = causal_mask & window_mask                  # (T, T)

        # Full attention matrix — for positions outside the window, set to -inf
        scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        scores = scores.masked_fill(~attend_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                       # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(out))


def visualize_attention_pattern(T: int = 16, W: int = 4):
    """Print the sliding window attention mask as ASCII."""
    row = torch.arange(T).unsqueeze(1)
    col = torch.arange(T).unsqueeze(0)
    mask = (col <= row) & (col >= (row - W + 1))
    print(f"Sliding window (T={T}, W={W}):")
    for i in range(T):
        row_str = "".join("■" if mask[i, j] else "·" for j in range(T))
        print(f"  {i:2d}: {row_str}")


if __name__ == "__main__":
    visualize_attention_pattern(16, 4)

    attn = SlidingWindowAttention(n_embd=128, n_head=4, window_size=8)
    x = torch.randn(2, 32, 128)
    out = attn(x)
    print(f"\nOutput shape: {out.shape}")
