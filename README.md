# AI Researcher Pathway — LLM Efficiency & Architecture

A structured, hands-on curriculum for becoming an LLM researcher.
Every project is implemented **from scratch in PyTorch** — no HuggingFace model classes, no black boxes.

---

## Philosophy

> "I do not know what I cannot build." — Feynman

Reading papers is not enough. You must implement every idea yourself, measure it, break it, and fix it. This repo is that process.

---

## Roadmap

### Phase 1 — Foundations
Build the core transformer stack from first principles.

| # | Project | Key Concept |
|---|---------|-------------|
| 01 | Bigram Language Model | N-gram statistics, token prediction |
| 02 | GPT-style Transformer | Causal attention, autoregressive LM |
| 03 | BERT-style Encoder | Masked LM, bidirectional attention |

### Phase 2 — Attention Mechanisms
Modern attention variants that power today's LLMs.

| # | Project | Key Concept |
|---|---------|-------------|
| 04 | Multi-Query Attention (MQA) | Shared K/V heads, KV bandwidth reduction |
| 05 | Grouped Query Attention (GQA) | Interpolation between MHA and MQA |
| 06 | Flash Attention (simplified) | Tiling, IO-awareness, memory efficiency |
| 07 | Linear Attention | O(n) complexity, kernel approximation |
| 08 | Sliding Window Attention | Local context, Longformer-style |

### Phase 3 — Positional Encodings
How transformers understand sequence order.

| # | Project | Key Concept |
|---|---------|-------------|
| 09 | Sinusoidal vs Learned PE | Absolute position, extrapolation |
| 10 | RoPE (Rotary PE) | Relative position via rotation matrices |
| 11 | ALiBi | Linear bias, length extrapolation |

### Phase 4 — Architecture Improvements
Innovations that made LLMs faster and better.

| # | Project | Key Concept |
|---|---------|-------------|
| 12 | Mixture of Experts (MoE) | Sparse routing, Switch Transformer |
| 13 | Parallel Transformer (PaLM-style) | Attention + FFN in parallel |
| 14 | SwiGLU + RMSNorm | Modern activation and normalization |

### Phase 5 — Inference Efficiency
Making models faster at generation time.

| # | Project | Key Concept |
|---|---------|-------------|
| 15 | KV Cache | Avoiding recomputation during decoding |
| 16 | Speculative Decoding | Draft-verify speedup |
| 17 | Post-Training Quantization | INT8/INT4, weight quantization |
| 18 | Knowledge Distillation | Teacher-student compression |

### Phase 6 — Parameter-Efficient Fine-Tuning (PEFT)
Adapting large models cheaply.

| # | Project | Key Concept |
|---|---------|-------------|
| 19 | LoRA | Low-rank weight updates |
| 20 | QLoRA | Quantized LoRA, 4-bit fine-tuning |
| 21 | Prefix Tuning | Soft prompt prepending |

### Phase 7 — Training Efficiency
Scaling training without scaling cost.

| # | Project | Key Concept |
|---|---------|-------------|
| 22 | Gradient Checkpointing | Memory vs compute tradeoff |
| 23 | Mixed Precision Training (AMP) | FP16/BF16, loss scaling |
| 24 | LR Schedulers | Cosine decay, warmup, WSD |

---

## Structure

```
AI_Researcher_Pathway/
├── phase1_foundations/
│   ├── 01_bigram_lm/
│   ├── 02_gpt_transformer/
│   └── 03_bert_encoder/
├── phase2_attention/
│   ├── 04_multi_query_attention/
│   ├── 05_grouped_query_attention/
│   ├── 06_flash_attention/
│   ├── 07_linear_attention/
│   └── 08_sliding_window_attention/
├── phase3_positional_encoding/
│   ├── 09_sinusoidal_vs_learned/
│   ├── 10_rope/
│   └── 11_alibi/
├── phase4_architecture/
│   ├── 12_mixture_of_experts/
│   ├── 13_parallel_transformer/
│   └── 14_swiglu_rmsnorm/
├── phase5_inference_efficiency/
│   ├── 15_kv_cache/
│   ├── 16_speculative_decoding/
│   ├── 17_quantization/
│   └── 18_distillation/
├── phase6_peft/
│   ├── 19_lora/
│   ├── 20_qlora/
│   └── 21_prefix_tuning/
└── phase7_training_efficiency/
    ├── 22_gradient_checkpointing/
    ├── 23_mixed_precision/
    └── 24_lr_schedulers/
```

## Setup

```bash
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10+
- PyTorch 2.x
- Basic calculus and linear algebra
- Familiarity with backpropagation

## Papers per Phase

Each project's folder contains a `PAPER.md` with the original paper reference and key equations explained.
