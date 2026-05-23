"""
GPT-style Autoregressive Transformer — from scratch.

Architecture:
  Token Embedding + Positional Embedding
  → N × TransformerBlock (CausalSelfAttention + FFN)
  → LayerNorm
  → LM Head (tied weights with token embedding)

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size:   int   = 65      # character-level tiny shakespeare default
    block_size:   int   = 256     # max sequence length
    n_embd:       int   = 384     # embedding dimension
    n_head:       int   = 6       # number of attention heads
    n_layer:      int   = 6       # number of transformer blocks
    dropout:      float = 0.2


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    For each position t, attention is only allowed to positions 0..t.
    This is enforced via an additive mask that sets future positions to -inf
    before softmax, making their contribution exactly zero.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        # Project input to Q, K, V in one shot
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,     bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of ones
        # Registered as a buffer so it moves with .to(device) automatically
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels

        # Compute Q, K, V and split heads
        qkv = self.c_attn(x)                    # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2) # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Weighted sum of values
        out = att @ v                            # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_drop(self.c_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise FFN: two linear layers with GELU in between.
    The inner dimension is 4× the embedding dimension (standard GPT ratio).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: LayerNorm BEFORE attention/FFN (not after).
    Pre-norm training is more stable — GPT-2 and all modern LLMs use it.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # residual connection
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(config.vocab_size, config.n_embd),
            pos_emb = nn.Embedding(config.block_size, config.n_embd),
            drop    = nn.Dropout(config.dropout),
            blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f    = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share token embedding with lm_head
        # This halves vocab parameters and improves performance
        self.transformer.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print(f"GPT initialized | params: {self.num_params():,}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        device = idx.device
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        tok = self.transformer.tok_emb(idx)     # (B, T, n_embd)
        pos = self.transformer.pos_emb(positions)  # (1, T, n_embd)
        x = self.transformer.drop(tok + pos)

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                # Zero out all logits below the top-k threshold
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
