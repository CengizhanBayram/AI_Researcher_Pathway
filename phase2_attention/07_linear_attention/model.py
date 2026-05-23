"""
Linear Attention — O(n) complexity via kernel feature maps — from scratch.

Standard softmax attention is O(n²d). Linear attention approximates it as:
  Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))

where φ is a feature map that approximates exp(x·y/√d).

The key trick: matrix association lets us compute (φ(K)^T V) first → O(nd²),
then multiply by φ(Q) → O(nd²), total O(nd²) instead of O(n²d).

We implement:
  1. Vanilla Linear Attention (φ(x) = elu(x) + 1)
  2. Performer: Random Fourier Features approximation

References:
  "Transformers are RNNs" — Katharopoulos et al., 2020 (arxiv.org/abs/2006.16236)
  "Rethinking Attention with Performers" — Choromanski et al., 2020 (arxiv.org/abs/2009.14794)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Simple feature map: φ(x) = elu(x) + 1
    Ensures all values are positive (required for linear attention to be valid).
    """
    return F.elu(x) + 1


def random_fourier_features(x: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    FAVOR+ feature map for Performers.
    Approximates exp(x·y/√d) using random orthogonal projections.

    φ(x) = exp(-||x||²/2) * [exp(ω_i · x)] for random ω_i ~ N(0, I)

    projection: (d, num_features) — random Gaussian matrix
    """
    d = x.shape[-1]
    xp = x @ projection / math.sqrt(d)                        # (..., num_features)
    norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2          # (..., 1)
    return torch.exp(xp - norm_sq) / math.sqrt(projection.shape[1])


class LinearAttention(nn.Module):
    """
    Causal linear attention using the ELU feature map.
    O(n·d²) time and O(d²) space (no T×T matrix).

    For causal (autoregressive) setting, we use the cumulative sum trick:
      S_i = Σ_{j≤i} φ(k_j)^T v_j    (d × d matrix, updated online)
      z_i = Σ_{j≤i} φ(k_j)           (d vector)
      output_i = φ(q_i) S_i / (φ(q_i) z_i)
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.n_embd   = n_embd

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q = self.W_q(x).view(B, T, H, D)  # (B, T, H, D)
        k = self.W_k(x).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)

        # Apply feature map
        q = elu_feature_map(q)
        k = elu_feature_map(k)

        out = torch.zeros_like(q)
        # Causal: iterate over time steps, maintain running S and z
        # S: (B, H, D, D)   z: (B, H, D)
        S = torch.zeros(B, H, D, D, device=x.device)
        z = torch.zeros(B, H, D,    device=x.device)

        for t in range(T):
            kt = k[:, t, :, :]  # (B, H, D)
            vt = v[:, t, :, :]  # (B, H, D)
            qt = q[:, t, :, :]  # (B, H, D)

            # Update: S += k^T v (outer product for each head)
            S = S + torch.einsum("bhd,bhe->bhde", kt, vt)
            z = z + kt

            # Compute output: φ(q) S / φ(q) z
            num = torch.einsum("bhd,bhde->bhe", qt, S)  # (B, H, D)
            den = torch.einsum("bhd,bhd->bh", qt, z).unsqueeze(-1).clamp(min=1e-6)  # (B, H, 1)
            out[:, t, :, :] = num / den

        out = out.reshape(B, T, C)
        return self.drop(self.W_o(out))


class PerformerAttention(nn.Module):
    """
    Full (non-causal) attention via FAVOR+ random feature approximation.
    For bidirectional models (BERT-style).
    """

    def __init__(self, n_embd: int, n_head: int, num_features: int = 256, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head      = n_head
        self.head_dim    = n_embd // n_head
        self.num_features = num_features

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

        # Fixed random projection (not learned)
        self.register_buffer(
            "projection",
            torch.randn(self.head_dim, num_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        F_ = self.num_features

        q = self.W_q(x).view(B, T, H, D)
        k = self.W_k(x).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)

        # Apply FAVOR+ feature map: (B, T, H, F_)
        qf = random_fourier_features(q, self.projection)
        kf = random_fourier_features(k, self.projection)

        # Linear attention: O = qf (kf^T v) / (qf kf^T 1)
        # kf^T v: (B, H, F_, D)
        kv = torch.einsum("bthf,bthd->bhfd", kf, v)
        # numerator: (B, T, H, D)
        num = torch.einsum("bthf,bhfd->bthd", qf, kv)
        # denominator: (B, T, H)
        ksum = kf.sum(dim=1)  # (B, H, F_)
        den  = torch.einsum("bthf,bhf->bth", qf, ksum).unsqueeze(-1).clamp(min=1e-6)

        out = (num / den).reshape(B, T, C)
        return self.drop(self.W_o(out))


if __name__ == "__main__":
    x = torch.randn(2, 128, 256)
    la = LinearAttention(n_embd=256, n_head=8)
    out = la(x)
    print(f"LinearAttention output: {out.shape}")  # (2, 128, 256)

    pa = PerformerAttention(n_embd=256, n_head=8, num_features=128)
    out2 = pa(x)
    print(f"PerformerAttention output: {out2.shape}")
