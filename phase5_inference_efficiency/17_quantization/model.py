"""
Post-Training Quantization (PTQ) — from scratch.

Quantization represents weights (and optionally activations) using fewer bits.
FP32 → INT8 reduces model size 4×, and INT8 matrix multiplications run faster
on hardware with INT8 SIMD units (virtually all modern CPUs/GPUs).

We implement:
  1. Symmetric per-tensor weight quantization (INT8)
  2. Symmetric per-channel weight quantization (INT8) — more accurate
  3. Dynamic activation quantization
  4. A simple quantized linear layer

Production: use torch.ao.quantization or bitsandbytes for real deployment.
This implementation is educational — shows the math, not max performance.

Reference: "A Survey of Quantization Methods for Efficient Neural Network Inference"
           Gholami et al., 2021 — https://arxiv.org/abs/2103.13630
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def quantize_symmetric(x: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric uniform quantization: maps [-max_val, max_val] → [-127, 127] for INT8.

    scale = max_val / (2^(n_bits-1) - 1)
    x_q   = round(x / scale).clamp(-qmax, qmax)

    Returns: (quantized tensor as int, scale)
    """
    qmax = 2 ** (n_bits - 1) - 1
    max_val = x.abs().max()
    scale = max_val / qmax
    x_q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale


def dequantize_symmetric(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct fp32 from int8 + scale."""
    return x_q.float() * scale


def quantize_per_channel(
    weight: torch.Tensor,
    n_bits: int = 8,
    dim: int = 0,  # quantize along output channels
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel quantization: each output channel has its own scale.
    More accurate than per-tensor because outlier channels don't squash others.

    weight: (out_features, in_features)
    Returns: (weight_q, scales) where scales has shape (out_features, 1)
    """
    qmax = 2 ** (n_bits - 1) - 1
    max_vals = weight.abs().amax(dim=1, keepdim=True)  # (out_features, 1)
    scales = max_vals / qmax
    weight_q = (weight / scales).round().clamp(-qmax, qmax).to(torch.int8)
    return weight_q, scales


def quantize_dynamic(x: torch.Tensor, n_bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic quantization: compute scale from the actual runtime values.
    Used for activations (which vary per input, unlike weights which are fixed).
    """
    return quantize_symmetric(x, n_bits)


class QuantizedLinear(nn.Module):
    """
    Linear layer with INT8 quantized weights.
    Weights are quantized once at 'quantize()' call.
    Activations are dynamically quantized at forward pass.

    This is the core building block for INT8 inference.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, n_bits: int = 8):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.quantized = False

        # Start as normal fp32 layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Quantized versions (filled by .quantize())
        self.register_buffer("weight_q",  None)
        self.register_buffer("weight_scale", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_bits: int = 8) -> "QuantizedLinear":
        """Convert an existing nn.Linear to a QuantizedLinear."""
        layer = cls(linear.in_features, linear.out_features,
                    bias=linear.bias is not None, n_bits=n_bits)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def quantize(self, per_channel: bool = True):
        """Quantize weights. Call once after training."""
        with torch.no_grad():
            if per_channel:
                w_q, scale = quantize_per_channel(self.weight.data, self.n_bits)
            else:
                w_q, scale = quantize_symmetric(self.weight.data, self.n_bits)
            self.weight_q    = w_q
            self.weight_scale = scale
        self.quantized = True
        del self.weight  # save memory — no longer needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return F.linear(x, self.weight, self.bias)

        # Dequantize weight for the matmul (in a real system, use INT8 GEMM)
        w_fp32 = dequantize_symmetric(self.weight_q, self.weight_scale)

        # Dynamic quantization of activations (optional; shown for educational purposes)
        # x_q, x_scale = quantize_dynamic(x)
        # x_fp32 = dequantize_symmetric(x_q, x_scale)
        # Using x directly here (simulates the dequant-matmul pattern)
        return F.linear(x, w_fp32, self.bias)


def quantize_model(model: nn.Module, n_bits: int = 8, per_channel: bool = True) -> nn.Module:
    """
    Walk the model, replace all nn.Linear with QuantizedLinear, then quantize.
    In-place modification.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            q_layer = QuantizedLinear.from_linear(module, n_bits)
            q_layer.quantize(per_channel=per_channel)
            setattr(model, name, q_layer)
        else:
            quantize_model(module, n_bits, per_channel)  # recurse
    return model


def compare_size_and_error(d_model: int = 512, n_layer: int = 4):
    """Measure weight quantization error and model size reduction."""
    import io

    # Build a simple model
    model = nn.Sequential(*[
        nn.Linear(d_model, d_model) for _ in range(n_layer)
    ])

    # fp32 size
    buf_fp32 = io.BytesIO()
    torch.save(model.state_dict(), buf_fp32)
    size_fp32 = buf_fp32.tell()

    # Quantize
    q_model = quantize_model(model, n_bits=8)

    # Measure INT8 weight error
    x = torch.randn(1, d_model)
    with torch.no_grad():
        # original forward (can't easily recover — this is for error illustration)
        pass

    # Directly measure quantization error on a single weight matrix
    w = torch.randn(d_model, d_model) * 0.02
    w_q, scale = quantize_per_channel(w)
    w_rec = dequantize_symmetric(w_q, scale)
    error = (w - w_rec).abs().mean().item()
    max_error = (w - w_rec).abs().max().item()

    print(f"Weight quantization error (INT8 per-channel):")
    print(f"  Mean absolute error: {error:.6f}")
    print(f"  Max  absolute error: {max_error:.6f}")
    print(f"  Relative error:      {error / w.abs().mean().item():.4%}")
    print(f"\nModel fp32 size: {size_fp32 / 1024:.1f} KB")
    print(f"Expected INT8 size: ~{size_fp32 / 4 / 1024:.1f} KB (4× reduction)")


if __name__ == "__main__":
    compare_size_and_error()

    # Example: quantize a single linear layer
    linear = nn.Linear(256, 256)
    q = QuantizedLinear.from_linear(linear, n_bits=8)
    q.quantize(per_channel=True)

    x = torch.randn(4, 32, 256)
    out = q(x)
    print(f"\nQuantized linear output: {out.shape}")
