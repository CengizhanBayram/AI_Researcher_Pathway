"""
Mixture of Experts (MoE) — from scratch.

In a standard transformer, every token passes through the same FFN.
In MoE, there are E expert FFNs, and a learned router selects the top-k
experts for each token. Only the selected experts compute, so the
model has more parameters but similar FLOPs.

Architecture (Switch Transformer style):
  - E expert FFNs (each identical in structure to standard FFN)
  - Router: linear layer → softmax → top-k selection
  - Load balancing loss: encourages uniform expert utilization

Used in: Switch Transformer, Mixtral 8×7B, DeepSeek-MoE, GPT-4 (rumored)

Reference: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
           Fedus et al., 2021 — https://arxiv.org/abs/2101.03961
           "Mixtral of Experts" — Jiang et al., 2024 — https://arxiv.org/abs/2401.04088
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MoEConfig:
    n_embd:          int   = 256
    n_experts:       int   = 8
    top_k:           int   = 2     # Mixtral uses top_k=2; Switch Transformer uses top_k=1
    ffn_mult:        int   = 4
    dropout:         float = 0.1
    load_balance_coeff: float = 0.01  # weight of auxiliary load balancing loss


class Expert(nn.Module):
    """A single FFN expert — identical to the standard transformer FFN."""

    def __init__(self, n_embd: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffn_mult * n_embd, bias=False),
            nn.SiLU(),  # SwiGLU-lite: SiLU instead of SiLU*gate
            nn.Linear(ffn_mult * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """
    Learns to assign tokens to experts.

    Outputs:
      router_logits: (B*T, E) — raw scores before softmax
      top_k_indices: (B*T, k) — which experts to use
      top_k_weights: (B*T, k) — softmax weights for the selected experts
    """

    def __init__(self, n_embd: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate  = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)                         # (N, E)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # (N, k)
        weights = F.softmax(weights, dim=-1)          # normalize selected weights
        return logits, indices, weights


class MoELayer(nn.Module):
    """
    MoE layer: router + E experts + load balancing loss.

    Token dispatch:
      For each token, compute routing weights for top-k experts.
      For each selected expert, gather all tokens routed to it, run
      the expert, then scatter the outputs back.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k     = config.top_k
        self.lb_coeff  = config.load_balance_coeff

        self.router  = Router(config.n_embd, config.n_experts, config.top_k)
        self.experts = nn.ModuleList([
            Expert(config.n_embd, config.ffn_mult, config.dropout)
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # (N, C)
        N = x_flat.shape[0]

        router_logits, expert_indices, expert_weights = self.router(x_flat)
        # expert_indices: (N, k)  expert_weights: (N, k)

        output = torch.zeros_like(x_flat)  # (N, C)

        for k_idx in range(self.top_k):
            # For each expert slot k, find which tokens chose expert e
            chosen_experts = expert_indices[:, k_idx]  # (N,)
            chosen_weights = expert_weights[:, k_idx]  # (N,)

            for e in range(self.n_experts):
                token_mask = (chosen_experts == e)  # (N,) bool
                if not token_mask.any():
                    continue

                tokens_for_expert = x_flat[token_mask]          # (n_e, C)
                expert_out = self.experts[e](tokens_for_expert)  # (n_e, C)

                # Weight and accumulate
                w = chosen_weights[token_mask].unsqueeze(-1)     # (n_e, 1)
                output[token_mask] += w * expert_out

        # Load balancing auxiliary loss (Switch Transformer Eq. 4-5)
        # Encourages equal token distribution across experts
        # f_i = fraction of tokens routed to expert i
        # P_i = average router probability for expert i
        # loss = n_experts * sum(f_i * P_i)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)
        # one-hot for top-1 for simplicity (full implementation counts all top-k)
        top1_indices = expert_indices[:, 0]
        tokens_per_expert = torch.bincount(top1_indices, minlength=self.n_experts).float()
        f = tokens_per_expert / N                              # (E,)
        P = router_probs.mean(dim=0)                           # (E,)
        load_balance_loss = self.n_experts * (f * P).sum()

        return output.view(B, T, C), self.lb_coeff * load_balance_loss


class MoETransformerBlock(nn.Module):
    """Transformer block where the FFN is replaced by a MoE layer."""

    def __init__(self, n_embd: int, n_head: int, config: MoEConfig, block_size: int = 512):
        super().__init__()
        from ..phase1_foundations.model import CausalSelfAttention, GPTConfig
        # Use the standard attention from project 02
        # (In practice, just inline or import your own attention)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # Inline attention to avoid cross-project imports
        self.attn = _InlineAttention(n_embd, n_head, block_size)
        self.moe  = MoELayer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, lb_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, lb_loss


class _InlineAttention(nn.Module):
    """Minimal causal attention (inline to avoid cross-project imports)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.c_attn   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj   = nn.Linear(n_embd, n_embd,     bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1) @ v
        return self.c_proj(att.transpose(1, 2).contiguous().view(B, T, C))


def expert_utilization_demo():
    """Show how the router distributes tokens across experts."""
    config = MoEConfig(n_embd=64, n_experts=4, top_k=1)
    moe = MoELayer(config)
    x = torch.randn(2, 16, 64)
    out, lb_loss = moe(x)
    print(f"Output: {out.shape}")
    print(f"Load balance loss: {lb_loss.item():.4f}")
    print(f"(Lower is better — perfect balance = 1.0 × lb_coeff = {config.load_balance_coeff})")


if __name__ == "__main__":
    expert_utilization_demo()
