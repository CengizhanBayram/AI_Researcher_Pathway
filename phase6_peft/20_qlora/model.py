"""
QLoRA (Quantized LoRA) — from scratch.

QLoRA = INT4 quantized base weights + LoRA adapters in BF16.

The base model is loaded in 4-bit NF4 (Normal Float 4) quantization.
LoRA adapters are still in full precision. This lets you fine-tune a
70B model on a single 48GB GPU (vs 140GB needed for full BF16 fine-tuning).

Innovations in the paper:
  1. NF4 (Normal Float 4): optimal quantization for normally-distributed weights
  2. Double quantization: quantize the quantization constants themselves
  3. Paged optimizers: handle memory spikes during backprop

This implementation shows the core ideas: NF4 quantization + LoRA integration.
For real training, use `bitsandbytes` library (implements the CUDA kernels).

Reference: "QLoRA: Efficient Finetuning of Quantized LLMs"
           Dettmers et al., 2023 — https://arxiv.org/abs/2305.14314
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# NF4 quantization levels — optimal for normally-distributed weights
# Derived from the quantiles of N(0,1) divided into 16 equal-probability buckets
NF4_QUANTIZATION_TABLE = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
], dtype=torch.float32)


def quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NF4 quantization: store each weight as 4-bit index into NF4 table.

    1. Divide weight into blocks of `block_size` elements
    2. For each block: normalize to [-1, 1] using absmax scaling
    3. Find nearest NF4 level (4-bit index, 0-15)

    Returns:
        indices:   (n_blocks * block_size // 2,) — packed 2×4bit per byte (uint8)
        absmax:    (n_blocks,) — per-block scaling factors
    """
    orig_shape = weight.shape
    weight_flat = weight.flatten()
    n = weight_flat.numel()

    # Pad to multiple of block_size
    pad = (block_size - n % block_size) % block_size
    if pad:
        weight_flat = F.pad(weight_flat, (0, pad))

    blocks = weight_flat.view(-1, block_size)           # (n_blocks, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True)     # (n_blocks, 1)
    normalized = blocks / absmax.clamp(min=1e-10)        # (n_blocks, block_size)

    # Find nearest NF4 level for each element
    table = NF4_QUANTIZATION_TABLE.to(weight.device)
    diff = (normalized.unsqueeze(-1) - table.view(1, 1, -1)).abs()
    indices = diff.argmin(dim=-1).to(torch.uint8)        # (n_blocks, block_size), values 0-15

    # Pack two 4-bit values into one byte (to actually save memory)
    # (n_blocks, block_size) → (n_blocks, block_size//2)
    high = indices[:, 0::2] << 4   # upper nibble
    low  = indices[:, 1::2]         # lower nibble
    packed = (high | low)           # (n_blocks, block_size//2)

    return packed.flatten(), absmax.squeeze(1), orig_shape, n


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, orig_shape: tuple, n_orig: int,
                   block_size: int = 64) -> torch.Tensor:
    """Reconstruct fp32 weights from NF4 quantized representation."""
    table = NF4_QUANTIZATION_TABLE.to(packed.device)
    n_blocks = absmax.shape[0]

    # Unpack bytes → two 4-bit indices each
    packed_2d = packed.view(n_blocks, -1)  # (n_blocks, block_size//2)
    high = (packed_2d >> 4) & 0xF          # upper nibble
    low  = packed_2d & 0xF                  # lower nibble

    # Interleave: (n_blocks, block_size)
    indices = torch.zeros(n_blocks, block_size, dtype=torch.long, device=packed.device)
    indices[:, 0::2] = high.long()
    indices[:, 1::2] = low.long()

    # Look up NF4 values
    values = table[indices]                # (n_blocks, block_size)

    # Rescale by absmax
    values = values * absmax.unsqueeze(1)

    # Remove padding and reshape
    values = values.flatten()[:n_orig]
    return values.view(orig_shape)


class NF4Linear(nn.Module):
    """
    Linear layer with NF4 quantized weights + LoRA adapters (QLoRA).

    Base weight is stored in NF4 (4-bit). LoRA adapters are in fp32/bf16.
    During forward: dequantize W to fp32, compute Wx + BAx.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        r:            int   = 8,
        alpha:        float = 16.0,
        block_size:   int   = 64,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r          = r
        self.scaling    = alpha / r
        self.block_size = block_size

        # LoRA adapters (trainable, in fp32)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Quantized weight (not a Parameter — no gradient)
        self.register_buffer("weight_packed", None)
        self.register_buffer("weight_absmax",  None)
        self._weight_orig_shape = None
        self._weight_n_orig     = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, alpha: float = 16.0, block_size: int = 64) -> "NF4Linear":
        layer = cls(linear.in_features, linear.out_features, r=r, alpha=alpha, block_size=block_size)
        packed, absmax, orig_shape, n_orig = quantize_nf4(linear.weight.data, block_size)
        layer.weight_packed = packed
        layer.weight_absmax  = absmax
        layer._weight_orig_shape = orig_shape
        layer._weight_n_orig     = n_orig
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize base weight on the fly
        w = dequantize_nf4(
            self.weight_packed, self.weight_absmax,
            self._weight_orig_shape, self._weight_n_orig, self.block_size
        )
        base = F.linear(x, w)
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        return base + lora * self.scaling

    def memory_saved_vs_fp32(self) -> float:
        """How many bytes saved vs fp32 (4 bytes per element → 0.5 bytes in NF4)."""
        n = self.in_features * self.out_features
        fp32_bytes = n * 4
        nf4_bytes  = n * 0.5 + self.weight_absmax.numel() * 4  # 4 bits + absmax in fp32
        return fp32_bytes / nf4_bytes


if __name__ == "__main__":
    linear = nn.Linear(256, 256, bias=False)
    nn.init.normal_(linear.weight, std=0.02)

    # Quantize to NF4
    packed, absmax, shape, n = quantize_nf4(linear.weight.data)
    w_rec = dequantize_nf4(packed, absmax, shape, n)
    error = (linear.weight.data - w_rec).abs().mean().item()
    print(f"NF4 reconstruction error: {error:.6f}")

    # QLoRA layer
    ql = NF4Linear.from_linear(linear, r=8, alpha=16)
    x = torch.randn(2, 32, 256)
    out = ql(x)
    print(f"QLoRA output: {out.shape}")
    print(f"Memory vs FP32: {ql.memory_saved_vs_fp32():.1f}× savings")
    print(f"Trainable params (LoRA only): {ql.lora_A.numel() + ql.lora_B.numel():,}")
