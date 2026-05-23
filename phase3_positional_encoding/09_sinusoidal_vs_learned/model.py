"""
Sinusoidal vs Learned Positional Encodings — from scratch.

Both methods inject position information into token embeddings.
The question is: which generalizes better beyond training length?

Reference: "Attention Is All You Need" (sinusoidal) — Vaswani et al., 2017
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
      - Can extrapolate to positions > max_seq_len seen during training
        (with degraded performance)
      - Relative position is encoded: PE(pos+k) can be expressed as a
        linear function of PE(pos) — this is what motivated the design
      - Zero parameters
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()           # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                     # (d/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)

        self.register_buffer("pe", pe)  # not a parameter — no gradient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings (GPT-style).

    Each position index gets its own learned embedding vector.
    Properties:
      - Cannot generalize to positions > max_seq_len (out-of-range index)
      - Generally matches or beats sinusoidal within the training range
      - Costs max_seq_len × d_model parameters
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(x + self.pe(positions))


def compare_encodings(d_model: int = 64, seq_len: int = 100):
    """Visualize the structure of sinusoidal vs learned encodings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sinusoidal
    sin_enc = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=seq_len + 200)
    dummy = torch.zeros(1, seq_len, d_model)
    sin_pe = sin_enc.pe[0, :seq_len, :].detach().numpy()

    # Learned (random init — not trained)
    lear_enc = LearnedPositionalEncoding(d_model=d_model, max_seq_len=seq_len)
    lear_pe = lear_enc.pe.weight.detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(sin_pe.T, aspect="auto", cmap="RdBu")
    axes[0].set_title("Sinusoidal PE")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")

    axes[1].imshow(lear_pe.T, aspect="auto", cmap="RdBu")
    axes[1].set_title("Learned PE (random init)")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dimension")

    plt.tight_layout()
    plt.savefig("pe_comparison.png", dpi=120)
    print("Saved pe_comparison.png")


if __name__ == "__main__":
    sin = SinusoidalPositionalEncoding(d_model=64)
    lea = LearnedPositionalEncoding(d_model=64, max_seq_len=512)

    x = torch.randn(2, 32, 64)
    print(f"Sinusoidal output: {sin(x).shape}")
    print(f"Learned output:    {lea(x).shape}")
    print(f"Sinusoidal params: {sum(p.numel() for p in sin.parameters())}")
    print(f"Learned params:    {sum(p.numel() for p in lea.parameters())}")

    compare_encodings()
