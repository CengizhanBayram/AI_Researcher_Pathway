"""
KV Cache — from scratch.

During autoregressive generation, naive attention recomputes K and V for
all previous tokens at every new step.

With KV cache: store K and V from all previous steps, append the new K, V,
and run attention only on the new token's query against the full cached KV.

Complexity:
  Without cache: O(T²) per token generated (recompute all)
  With cache:    O(T) per token generated

This is the single most impactful optimization for LLM inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCacheConfig:
    vocab_size:  int   = 65
    block_size:  int   = 512
    n_embd:      int   = 256
    n_head:      int   = 8
    n_layer:     int   = 4
    dropout:     float = 0.0   # 0 during inference


class KVCache:
    """
    Per-layer cache of Key and Value tensors.

    Grows one step at a time during generation.
    Shape: (batch_size, n_head, seq_len, head_dim)
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)  # append on seq dim
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def clear(self):
        self.k = None
        self.v = None

    @property
    def seq_len(self) -> int:
        return self.k.shape[2] if self.k is not None else 0


class CachedAttention(nn.Module):
    """
    Causal attention with optional KV cache.

    During prefill (processing the prompt): cache=None → standard full attention
    During generation (one token at a time): cache=KVCache → O(T) per step
    """

    def __init__(self, config: KVCacheConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd   = config.n_embd

        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)  # k, v now contain full history

        T_full = k.shape[2]  # total sequence length including cache

        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T_full)

        if cache is None:
            # Standard causal mask during prefill
            att = att.masked_fill(self.mask[:, :, :T, :T_full] == 0, float("-inf"))
        # During generation (T=1), no masking needed — single query attends to all past

        att = F.softmax(att, dim=-1)
        out = att @ v                                              # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class CachedTransformerBlock(nn.Module):
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.attn = CachedAttention(config)
        self.ffn  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache)
        x = x + self.ffn(self.ln2(x))
        return x


class GPTWithKVCache(nn.Module):
    def __init__(self, config: KVCacheConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks  = nn.ModuleList([CachedTransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        caches: Optional[list[KVCache]] = None,
        start_pos: int = 0,
    ):
        B, T = idx.shape
        device = idx.device
        positions = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(positions)

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = block(x, cache)

        logits = self.lm_head(self.ln_f(x))
        return logits

    @torch.no_grad()
    def generate_with_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        caches = [KVCache() for _ in self.blocks]

        # Prefill: process the prompt in one shot
        _ = self(idx, caches=caches, start_pos=0)
        start_pos = idx.shape[1]

        # Decode: one token at a time using cache
        for _ in range(max_new_tokens):
            last_token = idx[:, -1:]                     # (B, 1)
            logits = self(last_token, caches=caches, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_t], dim=1)
            start_pos += 1

        return idx

    @torch.no_grad()
    def generate_without_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_t], dim=1)
        return idx


def benchmark_cache(config: KVCacheConfig, prompt_len: int = 32, gen_len: int = 64):
    """Compare wall-clock time with and without KV cache."""
    import time
    model = GPTWithKVCache(config).eval()
    prompt = torch.zeros((1, prompt_len), dtype=torch.long)

    # Without cache
    t0 = time.perf_counter()
    _ = model.generate_without_cache(prompt, gen_len)
    t_no_cache = time.perf_counter() - t0

    # With cache
    t0 = time.perf_counter()
    _ = model.generate_with_cache(prompt, gen_len)
    t_cache = time.perf_counter() - t0

    print(f"Without KV cache: {t_no_cache:.3f}s")
    print(f"With    KV cache: {t_cache:.3f}s")
    print(f"Speedup: {t_no_cache / t_cache:.2f}×")


if __name__ == "__main__":
    config = KVCacheConfig(vocab_size=65, n_embd=128, n_head=4, n_layer=4, block_size=256)
    print("Benchmarking KV cache...")
    benchmark_cache(config, prompt_len=32, gen_len=64)
