"""
Gradient Checkpointing — from scratch.

Standard backpropagation stores ALL intermediate activations in memory.
For a transformer with T=2048 and N=32 layers, this is enormous.

Gradient checkpointing trades compute for memory:
  - During forward: only store activations at "checkpoint" boundaries
  - During backward: recompute intermediate activations from nearest checkpoint

Memory tradeoff:
  Standard:     O(N) activations for N layers
  Checkpointing: O(√N) activations — recompute the rest

Cost: ~33% more compute (one extra forward pass per checkpoint segment)

PyTorch has torch.utils.checkpoint.checkpoint() built-in, but we implement
it manually to understand the mechanism.

Reference: "Training Deep Nets with Sublinear Memory Cost"
           Chen et al., 2016 — https://arxiv.org/abs/1604.06174
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerBlock(nn.Module):
    """
    Transformer block that uses gradient checkpointing for its attention + FFN.

    When checkpointed=True: activations inside the block are NOT stored.
    During backward, the block's forward is re-run to reconstruct them.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int = 512,
                 dropout: float = 0.1, checkpointed: bool = True):
        super().__init__()
        self.checkpointed = checkpointed
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        self.W_k = nn.Linear(n_embd, n_embd, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )
        self.drop = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) * self.head_dim**-0.5
                        + self.mask[:, :, :T, :T].log(), dim=-1)
        return self.W_o((att @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return self.drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointed and self.training:
            # PyTorch's checkpoint: don't store activations, recompute in backward
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class GPTWithCheckpointing(nn.Module):
    """
    GPT with optional gradient checkpointing per block.
    Can checkpoint every block or every k-th block (partial checkpointing).
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int = 512, dropout: float = 0.1,
                 checkpoint_every: int = 1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList([
            CheckpointedTransformerBlock(
                n_embd, n_head, block_size, dropout,
                checkpointed=(i % checkpoint_every == 0),
            )
            for i in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


def measure_memory(model_fn, x, y, label: str):
    """Measure peak GPU memory during a training step."""
    if not torch.cuda.is_available():
        print(f"{label}: (CUDA not available, skipping memory measurement)")
        return

    torch.cuda.reset_peak_memory_stats()
    model = model_fn().cuda().train()
    x, y = x.cuda(), y.cuda()
    logits, loss = model(x, y)
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"{label}: peak memory = {peak_mb:.1f} MB")


if __name__ == "__main__":
    # Demonstrate the API
    model = GPTWithCheckpointing(
        vocab_size=65, n_embd=256, n_head=8, n_layer=12, block_size=256,
        checkpoint_every=1  # checkpoint every block
    )
    model.train()
    x = torch.randint(0, 65, (4, 128))
    y = torch.randint(0, 65, (4, 128))
    _, loss = model(x, y)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Checkpointed blocks: {sum(1 for b in model.blocks if b.checkpointed)}/{len(model.blocks)}")
