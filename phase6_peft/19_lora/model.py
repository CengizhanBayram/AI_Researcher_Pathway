"""
LoRA (Low-Rank Adaptation) — from scratch.

Full fine-tuning updates all W ∈ R^{d×k} weights. LoRA instead adds
a trainable low-rank update: ΔW = B·A where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k).

During fine-tuning: only A and B are trained (W is frozen).
During inference: merge ΔW into W → zero overhead.

Forward pass: h = Wx + (BA)x * (α/r)
  - α/r: a scaling factor (α is a hyperparameter, usually α = r or α = 2r)
  - B initialized to zeros, A initialized with Gaussian → ΔW = 0 at start

Parameter reduction example:
  d=1024, k=1024, r=8:
    Full: 1024×1024 = 1M params
    LoRA: 1024×8 + 8×1024 = 16K params (62× fewer)

Used in: Alpaca, Vicuna, LLaMA fine-tuning, stable diffusion

Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
           Hu et al., 2021 — https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with LoRA adapters.

    The original weight W is frozen. Only A and B are trained.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        r:            int   = 8,     # rank
        alpha:        float = 16.0,  # scaling (usually 2r)
        dropout:      float = 0.1,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r     = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # FROZEN

        self.bias = None  # simplified: no bias

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B=0 → ΔW=0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_drop = nn.Dropout(dropout)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.1) -> "LoRALinear":
        """Wrap an existing nn.Linear with LoRA."""
        layer = cls(linear.in_features, linear.out_features, r=r, alpha=alpha, dropout=dropout)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias = nn.Parameter(linear.bias.clone())
                layer.bias.requires_grad = False
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base (frozen)
        base_out = F.linear(x, self.weight, self.bias)

        if self.merged:
            return base_out

        # LoRA contribution: x → A → B, scaled
        lora_out = (self.lora_drop(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def merge(self):
        """Merge LoRA into base weight for zero-overhead inference."""
        if self.merged:
            return
        with torch.no_grad():
            delta_W = self.lora_B @ self.lora_A  # (out, r) @ (r, in) = (out, in)
            self.weight.data += delta_W * self.scaling
        self.merged = True

    def unmerge(self):
        """Unmerge (useful for switching between base and fine-tuned)."""
        if not self.merged:
            return
        with torch.no_grad():
            delta_W = self.lora_B @ self.lora_A
            self.weight.data -= delta_W * self.scaling
        self.merged = False

    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def total_params(self) -> int:
        return self.weight.numel() + self.lora_A.numel() + self.lora_B.numel()


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05,
                target_modules: set[str] | None = None) -> nn.Module:
    """
    Replace target Linear layers with LoRALinear.
    If target_modules is None, replaces all nn.Linear layers.
    Typical targets: {"W_q", "W_v"} (just Q and V projections, as in original paper)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or name in target_modules:
                lora_layer = LoRALinear.from_linear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(model, name, lora_layer)
        else:
            inject_lora(module, r, alpha, dropout, target_modules)
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def demo_lora():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../phase1_foundations/02_gpt_transformer"))
    from model import GPT, GPTConfig

    config = GPTConfig(vocab_size=65, block_size=128, n_embd=256, n_head=8, n_layer=4)
    model  = GPT(config)

    trainable_before, total_before = count_trainable_params(model)
    print(f"Before LoRA: {trainable_before:,} trainable / {total_before:,} total")

    # Freeze all base parameters
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA into Q and V projections of all attention layers
    inject_lora(model, r=8, alpha=16, target_modules={"W_q", "W_v"})

    trainable_after, total_after = count_trainable_params(model)
    print(f"After  LoRA: {trainable_after:,} trainable / {total_after:,} total")
    print(f"Trainable %: {100 * trainable_after / total_after:.2f}%")

    x = torch.randint(0, 65, (2, 64))
    logits, _ = model(x)
    print(f"Output shape: {logits.shape}")


if __name__ == "__main__":
    demo_lora()
