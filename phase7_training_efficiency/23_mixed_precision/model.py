"""
Mixed Precision Training (AMP) — from scratch.

FP32: 32-bit floats, ~6 digits precision, 4 bytes/element
FP16: 16-bit floats, ~3 digits precision, 2 bytes/element
BF16: 16-bit floats, same exponent range as FP32, less mantissa precision

Mixed precision keeps weights in FP32 (the "master copy") for numerical stability,
but computes forward/backward in FP16/BF16 for speed and memory savings.

Key techniques:
  1. Loss scaling: multiply loss by a large scale before backward to prevent
     FP16 underflow of gradients (values < 1e-8 become 0 in FP16)
  2. Master weights in FP32: prevent weight underflow during optimizer steps
  3. GradScaler: automatically tune the loss scale

Speedup: 2-4× on modern GPUs with Tensor Cores (A100, H100, 4090)
Memory:  ~2× reduction in activation memory

Reference: "Mixed Precision Training" — Micikevicius et al., 2018
           https://arxiv.org/abs/1710.03740
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import contextmanager


class ManualLossScaler:
    """
    Manual loss scaling implementation — shows what PyTorch's GradScaler does internally.

    Algorithm:
      1. Multiply loss by scale S before backward
      2. Unscale gradients: grad /= S
      3. Check for inf/nan in gradients
      4. If clean: step optimizer, increase S gradually
      5. If inf/nan: skip optimizer step, decrease S
    """

    def __init__(self, init_scale: float = 2**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale          = init_scale
        self.growth_factor  = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_and_check(self, optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients and return True if all finite."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data /= self.scale
                if not torch.isfinite(p.grad).all():
                    return False
        return True

    def step(self, optimizer: torch.optim.Optimizer):
        """Unscale, check, step (or skip if overflow)."""
        all_finite = self.unscale_and_check(optimizer)
        if all_finite:
            optimizer.step()
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, 2**24)
                self._growth_tracker = 0
        else:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._growth_tracker = 0

    def zero_grad(self, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()


def train_fp32(model: nn.Module, optimizer, data_iter, n_steps: int, device: str) -> list[float]:
    """Standard FP32 training — baseline."""
    model = model.to(device).to(torch.float32)
    losses = []
    for step, (x, y) in enumerate(data_iter):
        if step >= n_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train_amp(model: nn.Module, optimizer, data_iter, n_steps: int, device: str,
              dtype: torch.dtype = torch.float16) -> list[float]:
    """
    AMP training using PyTorch's torch.amp.autocast + GradScaler.
    - autocast: automatically casts ops to fp16 where safe
    - GradScaler: handles loss scaling to prevent gradient underflow
    """
    model = model.to(device).to(torch.float32)  # master weights in fp32
    scaler = GradScaler()
    losses = []

    for step, (x, y) in enumerate(data_iter):
        if step >= n_steps:
            break
        x, y = x.to(device), y.to(device)

        # autocast region: forward pass and loss computation in fp16
        with autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()       # scale loss to prevent underflow
        scaler.unscale_(optimizer)           # unscale before clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)              # only steps if gradients are finite
        scaler.update()                     # adjust scale factor

        losses.append(loss.item())
    return losses


class BF16TrainingWrapper:
    """
    BF16 training (preferred over FP16 on Ampere+ GPUs — no loss scaling needed
    because BF16 has the same exponent range as FP32).
    """

    def __init__(self, model: nn.Module, optimizer, device: str = "cuda"):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.device    = device

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        with autocast(device_type=self.device, dtype=torch.bfloat16):
            _, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


def benchmark_precision(n_embd: int = 256, n_layer: int = 4, T: int = 256, B: int = 16):
    """Compare FP32, FP16 AMP, BF16 AMP training speed (requires CUDA)."""
    import time, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../phase1_foundations/02_gpt_transformer"))
    from model import GPT, GPTConfig

    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmark")
        return

    device = "cuda"
    cfg = GPTConfig(vocab_size=65, block_size=T, n_embd=n_embd, n_head=8, n_layer=n_layer)

    x = torch.randint(0, 65, (B, T))
    y = torch.randint(0, 65, (B, T))

    results = {}
    for dtype, label in [(torch.float32, "FP32"), (torch.float16, "FP16 AMP"), (torch.bfloat16, "BF16 AMP")]:
        model = GPT(cfg).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(5):
            with autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(x.to(device), y.to(device))
            loss.backward()
            opt.zero_grad()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            with autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(x.to(device), y.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        results[label] = elapsed

    baseline = results["FP32"]
    for label, t in results.items():
        print(f"{label:12s}: {t:.3f}s  ({baseline/t:.2f}× vs FP32)")


if __name__ == "__main__":
    print("Mixed precision training module loaded.")
    print("Call benchmark_precision() to compare FP32 vs AMP on GPU.")
    benchmark_precision()
