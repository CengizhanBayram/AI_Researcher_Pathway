# AI Researcher Pathway — LLM Efficiency & Architecture

A structured, hands-on curriculum for becoming an LLM researcher.
Every project is implemented **from scratch in PyTorch** — no HuggingFace model classes, no black boxes.

---

## Philosophy

> "I do not know what I cannot build." — Feynman

Reading papers is not enough. You must implement every idea yourself, measure it, break it, and fix it.

The pattern for each project:
1. Read the paper (linked in each `PAPER.md`)
2. Implement from scratch
3. Run the experiment and record numbers
4. Break one assumption and observe what happens
5. Read the follow-up papers

The goal is not to reproduce SOTA numbers — it is to build the intuition that lets you read a new paper and immediately know which part is the real contribution, which part is engineering, and which part is noise.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA optional (CPU works for all projects).

---

## Roadmap

### Phase 1 — Foundations

> Goal: understand exactly what a transformer is and why it works.

The right order matters here. Start with the bigram model — it will feel trivial, but it gives you the baseline loss you'll use to measure every future improvement. Then add attention layer by layer.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 01 | [Bigram Language Model](phase1_foundations/01_bigram_lm/) | A V×V embedding table — the simplest possible LM | Why does val loss plateau at ~2.5? What is the theoretical minimum? |
| 02 | [GPT Transformer](phase1_foundations/02_gpt_transformer/) | Full GPT: causal attention, pre-norm, weight tying, top-k sampling | How much does each component (attention vs FFN vs depth) contribute to the loss drop? |
| 03 | [BERT Encoder](phase1_foundations/03_bert_encoder/) | Bidirectional encoder, MLM masking strategy, NSP head | Why can't BERT be used directly for generation? What makes bidirectional attention unsuitable for autoregressive decoding? |

**Milestone:** After Phase 1, you should be able to train a GPT from scratch, reach val loss ~1.5 on tiny shakespeare, and generate coherent text. You should also understand exactly why MHA is O(T²) and why that becomes a problem at T=128K.

---

### Phase 2 — Attention Mechanisms

> Goal: understand the bottleneck that drives every modern LLM architectural decision.

The attention score matrix is O(T²) in memory. At T=32K (LLaMA 3), that's 32K² × 4 bytes × n_heads = hundreds of GB per layer. Every project in this phase attacks that problem from a different angle.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 04 | [Multi-Query Attention](phase2_attention/04_multi_query_attention/) | n Q heads share a single K and V head | How much does KV cache shrink? What is the quality tradeoff on your benchmark? |
| 05 | [Grouped Query Attention](phase2_attention/05_grouped_query_attention/) | G groups of Q heads share G pairs of K/V heads | Where on the MHA↔MQA quality-efficiency curve does GQA sit? Run the sweep. |
| 06 | [Flash Attention](phase2_attention/06_flash_attention/) | Tiled online-softmax — never materializes the T×T matrix | Verify numerical equivalence to naive attention. Measure peak memory at T=512, 1024, 2048. |
| 07 | [Linear Attention](phase2_attention/07_linear_attention/) | ELU feature map (O(n)) and Performer FAVOR+ approximation | At what sequence length does linear attention become faster than softmax attention? |
| 08 | [Sliding Window Attention](phase2_attention/08_sliding_window_attention/) | Each token attends to a local window of w positions | What window size is needed to match full attention quality on a language modeling task? |

**Core equation — standard attention:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```
Every project in this phase is a different approximation or restructuring of this equation.

**Milestone:** Implement MHA, MQA, and GQA in the same training loop, compare perplexities, and plot KV cache size vs quality. This is a publishable experiment at small scale.

---

### Phase 3 — Positional Encodings

> Goal: understand how position enters the model and why it is the main obstacle to long-context.

Transformers have no inherent notion of order — without positional information, "the dog bit the man" and "the man bit the dog" are identical. These three approaches encode position in fundamentally different ways.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 09 | [Sinusoidal vs Learned PE](phase3_positional_encoding/09_sinusoidal_vs_learned/) | Fixed sin/cos encoding + learned embedding table | Train on T=256, evaluate on T=512. Which generalizes better? |
| 10 | [RoPE](phase3_positional_encoding/10_rope/) | Rotation matrices applied to Q and K — encodes relative position in the dot product | Verify the relative-position property mathematically: score(q_m, k_n) = f(m-n). |
| 11 | [ALiBi](phase3_positional_encoding/11_alibi/) | Linear bias added to attention scores, no embedding at all | Train on T=512, test on T=2048. Measure perplexity degradation vs RoPE and sinusoidal. |

**Key insight — why RoPE won:**
Sinusoidal PE adds position to the token embedding (absolute). ALiBi adds a bias to attention scores (relative, but fixed). RoPE rotates Q and K vectors so that the dot product naturally encodes relative position — and it can be extended to longer contexts via scaling (YaRN, LongRoPE).

**Milestone:** Reproduce the length-extrapolation experiment from the ALiBi paper at small scale. Plot perplexity vs context length for all three methods.

---

### Phase 4 — Architecture Improvements

> Goal: understand the architectural choices that separate GPT-2 from LLaMA 3.

These are not incremental tweaks — each one changes the compute graph in a fundamental way.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 12 | [Mixture of Experts](phase4_architecture/12_mixture_of_experts/) | E expert FFNs + learned router, top-k dispatch, load balancing loss | How does the load balancing coefficient affect expert utilization? What happens without it? |
| 13 | [Parallel Transformer](phase4_architecture/13_parallel_transformer/) | Attention and FFN computed simultaneously on the same LN(x) | Does the parallel structure hurt convergence? Compare loss curves on the same dataset. |
| 14 | [SwiGLU + RMSNorm](phase4_architecture/14_swiglu_rmsnorm/) | SiLU-gated FFN (SwiGLU) + RMS-based normalization | Benchmark RMSNorm vs LayerNorm speed. Compare SwiGLU vs GELU FFN at the same parameter count. |

**The LLaMA 3 block vs GPT-2 block:**
```
GPT-2:    x → LayerNorm → MHA  → add  →  LayerNorm → GELU FFN → add
LLaMA 3:  x → RMSNorm  → RoPE MHA → add → RMSNorm → SwiGLU FFN → add
```
Projects 10, 12, and 14 together give you the LLaMA block. Project 05 (GQA) gives you the LLaMA 2 attention. That is the full modern LLM architecture.

**Milestone:** Build a LLaMA-style model by composing: GQA (05) + RoPE (10) + SwiGLU + RMSNorm (14). Train it on tiny shakespeare and compare loss curves to your plain GPT from project 02.

---

### Phase 5 — Inference Efficiency

> Goal: understand why inference is the hard problem, not training.

Training a 7B model costs ~$1M once. Serving it costs that every few weeks. The projects here are the core techniques that make deployment economically viable.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 15 | [KV Cache](phase5_inference_efficiency/15_kv_cache/) | Per-layer cache of K and V tensors, incremental decoding | Benchmark generation speed with/without cache at various context lengths. Plot tokens/second. |
| 16 | [Speculative Decoding](phase5_inference_efficiency/16_speculative_decoding/) | Small draft model proposes γ tokens; large target verifies in 1 pass | How does acceptance rate vary with draft model quality? Plot speedup vs γ. |
| 17 | [Post-Training Quantization](phase5_inference_efficiency/17_quantization/) | Symmetric per-tensor and per-channel INT8, quantized linear layer | Measure reconstruction error (per-tensor vs per-channel). How much does quantization error affect perplexity? |
| 18 | [Knowledge Distillation](phase5_inference_efficiency/18_distillation/) | Teacher–student training with temperature-scaled KL divergence | At the same parameter count, does a distilled student beat a student trained from scratch? |

**Numbers to have in your head:**

| Method | Memory reduction | Speedup | Quality loss |
|--------|-----------------|---------|--------------|
| KV Cache | none (required for usability) | 10-50× | none |
| INT8 Quant | 2× | 1.5-2× | < 0.5 ppl |
| INT4 Quant | 4× | 2-3× | 1-3 ppl |
| Speculative (γ=4) | none | 2-3× | none (exact) |
| Distillation 2× | 2× fewer params | 2× | depends on budget |

**Milestone:** Implement KV cache and measure the speedup at context lengths 64, 256, 1024. The speedup should grow with context length — understand why.

---

### Phase 6 — Parameter-Efficient Fine-Tuning (PEFT)

> Goal: adapt a pretrained model to a new task with 0.1% of the parameters.

Fine-tuning all 7B parameters for each downstream task is not feasible. These methods find the minimal trainable footprint that achieves competitive performance.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 19 | [LoRA](phase6_peft/19_lora/) | Low-rank matrices A and B inserted beside frozen weight W: ΔW = BA | How small can rank r be before quality collapses? Plot val loss vs r for r ∈ {1, 2, 4, 8, 16, 32}. |
| 20 | [QLoRA](phase6_peft/20_qlora/) | NF4 (4-bit Normal Float) quantized base weights + LoRA adapters in fp32 | Measure NF4 reconstruction error vs INT8 vs INT4 uniform quantization on the same weights. |
| 21 | [Prefix Tuning](phase6_peft/21_prefix_tuning/) | Learnable K/V prefix prepended to every attention layer, base model frozen | Compare prefix tuning vs LoRA at the same trainable parameter budget. Which wins and why? |

**Parameter counts (illustrative, GPT-2 Medium 345M):**

| Method | Trainable params | % of total |
|--------|-----------------|------------|
| Full fine-tuning | 345M | 100% |
| LoRA r=8 (Q,V only) | ~800K | 0.23% |
| Prefix tuning (T=20) | ~370K | 0.11% |
| QLoRA r=8 | ~800K + 4-bit base | 0.23% trainable |

**Milestone:** Freeze a pretrained GPT (from project 02), inject LoRA, fine-tune on a different text domain. Compare convergence speed and final loss to full fine-tuning.

---

### Phase 7 — Training Efficiency

> Goal: train larger models on the same hardware budget.

These techniques are what allow a research lab with 8 GPUs to train a model that would normally require 32 GPUs.

| # | Project | What you build | Key question to answer |
|---|---------|----------------|------------------------|
| 22 | [Gradient Checkpointing](phase7_training_efficiency/22_gradient_checkpointing/) | Recompute activations during backward instead of storing them | Measure peak memory with/without checkpointing for depth 4, 8, 16, 32 layers. |
| 23 | [Mixed Precision (AMP)](phase7_training_efficiency/23_mixed_precision/) | FP16/BF16 forward pass, FP32 master weights, GradScaler | Benchmark FP32 vs FP16 AMP vs BF16 AMP throughput (tokens/second). |
| 24 | [LR Schedulers](phase7_training_efficiency/24_lr_schedulers/) | Cosine+warmup (GPT-3 style), WSD (MiniCPM style) | Does WSD's stable phase allow you to extend training without restarting? Verify empirically. |

**Memory budget breakdown for a 1B parameter model, batch=1, T=2048:**

| Component | Memory (FP32) | Memory (BF16 AMP) |
|-----------|--------------|-------------------|
| Parameters | 4 GB | 2 GB |
| Gradients | 4 GB | 4 GB (kept in FP32) |
| Activations (no checkpointing) | ~8 GB | ~4 GB |
| Activations (with checkpointing) | ~1.5 GB | ~0.75 GB |
| Optimizer states (AdamW) | 8 GB | 8 GB (FP32) |
| **Total** | **~25 GB** | **~19 GB** |

**Milestone:** Train a 12-layer model with and without gradient checkpointing + AMP. Measure: peak memory, throughput (tokens/sec), and final loss. The memory-speed frontier is the core systems-ML tradeoff.

---

## Learning Path Connections

The projects are not independent. Here is how they connect:

```
01 Bigram
    └── 02 GPT (adds attention)
            ├── 03 BERT (removes causal mask)
            ├── 04 MQA → 05 GQA     (fix KV bandwidth)
            ├── 06 Flash             (fix memory usage)
            ├── 07 Linear            (fix quadratic complexity)
            ├── 08 Sliding Window    (fix long context)
            ├── 09/10/11 PE          (fix length generalization)
            ├── 12 MoE               (scale parameters cheaply)
            ├── 13 Parallel          (speed up compute)
            ├── 14 SwiGLU+RMSNorm    (improve stability)
            │
            ├── [LLaMA block = 02 + 05 + 10 + 14]
            │
            ├── 15 KV Cache          (fast inference)
            ├── 16 Speculative       (faster inference)
            ├── 17 Quantization      (smaller model)
            ├── 18 Distillation      (smaller model)
            │
            ├── 19 LoRA → 20 QLoRA  (cheap fine-tuning)
            ├── 21 Prefix Tuning     (alternative PEFT)
            │
            ├── 22 Grad Checkpoint  (train deeper)
            ├── 23 AMP              (train faster)
            └── 24 LR Schedulers    (train better)
```

---

## Paper Reading Order

For each phase, read these papers before or alongside the implementation:

**Phase 1**
- "Attention Is All You Need" — Vaswani et al., 2017
- "Language Models are Unsupervised Multitask Learners" (GPT-2) — Radford et al., 2019
- "BERT" — Devlin et al., 2018

**Phase 2**
- "Fast Transformer Decoding: One Write-Head is All You Need" (MQA) — Shazeer, 2019
- "GQA" — Ainslie et al., 2023
- "FlashAttention" — Dao et al., 2022
- "Transformers are RNNs" (Linear Attn) — Katharopoulos et al., 2020
- "Longformer" — Beltagy et al., 2020

**Phase 3**
- "Attention Is All You Need" (sinusoidal)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" — Su et al., 2021
- "Train Short, Test Long: ALiBi" — Press et al., 2021

**Phase 4**
- "Switch Transformers" — Fedus et al., 2021
- "Mixtral of Experts" — Jiang et al., 2024
- "PaLM" — Chowdhery et al., 2022
- "GLU Variants Improve Transformer" — Shazeer, 2020
- "Root Mean Square Layer Normalization" — Zhang & Sennrich, 2019

**Phase 5**
- "Efficient Inference" — Pope et al., 2022 (KV cache analysis)
- "Fast Inference via Speculative Decoding" — Leviathan et al., 2023
- "A Survey of Quantization Methods" — Gholami et al., 2021
- "Distilling the Knowledge in a Neural Network" — Hinton et al., 2015

**Phase 6**
- "LoRA: Low-Rank Adaptation of Large Language Models" — Hu et al., 2021
- "QLoRA: Efficient Finetuning of Quantized LLMs" — Dettmers et al., 2023
- "Prefix-Tuning: Optimizing Continuous Prompts for Generation" — Li & Liang, 2021

**Phase 7**
- "Training Deep Nets with Sublinear Memory Cost" — Chen et al., 2016
- "Mixed Precision Training" — Micikevicius et al., 2018
- "MiniCPM" (WSD scheduler) — Hu et al., 2024

---

## Experiments to Run (Research Mindset)

These are the experiments that go beyond just "make it work." Each one is a small research contribution if you document it properly.

1. **Attention head ablation:** Train the GPT, then zero out one head at a time. Which heads are expendable? Are some heads redundant?

2. **Positional encoding length extrapolation benchmark:** Train all three PE methods (sinusoidal, RoPE, ALiBi) on T=256, evaluate on T=512, 1024, 2048. Plot perplexity vs context length.

3. **MHA vs MQA vs GQA quality-efficiency frontier:** For a fixed FLOPs budget, which configuration gives the best perplexity? This is exactly the experiment in the GQA paper.

4. **LoRA rank sensitivity:** For a fixed total parameter budget, is it better to use LoRA on more layers with small r, or fewer layers with large r?

5. **KV cache memory vs context length:** Measure peak memory during generation as context length grows. At what length does the KV cache dominate over model weights?

6. **Distillation vs training from scratch:** At the same parameter count, does a student distilled from a 10× larger teacher beat a model trained from scratch? How many tokens of distillation data are needed to match scratch training?

---

## File Structure

```
AI_Researcher_Pathway/
├── README.md
├── requirements.txt
│
├── phase1_foundations/
│   ├── 01_bigram_lm/
│   │   ├── model.py          ← BigramLM class
│   │   ├── train.py          ← training loop
│   │   └── PAPER.md          ← paper links + key equations
│   ├── 02_gpt_transformer/
│   │   ├── model.py          ← GPTConfig, CausalSelfAttention, GPT
│   │   ├── train.py
│   │   └── PAPER.md
│   └── 03_bert_encoder/
│       ├── model.py          ← BERTConfig, BidirectionalSelfAttention, BERT, MLMHead
│       ├── train.py
│       └── PAPER.md
│
├── phase2_attention/
│   ├── 04_multi_query_attention/   ← MultiQueryAttention
│   ├── 05_grouped_query_attention/ ← GroupedQueryAttention
│   ├── 06_flash_attention/         ← flash_attention_tiled, verify_equivalence
│   ├── 07_linear_attention/        ← LinearAttention, PerformerAttention
│   └── 08_sliding_window_attention/← SlidingWindowAttention, visualize_attention_pattern
│
├── phase3_positional_encoding/
│   ├── 09_sinusoidal_vs_learned/  ← SinusoidalPE, LearnedPE, compare_encodings
│   ├── 10_rope/                   ← precompute_rope_frequencies, apply_rope, RoPESelfAttention
│   └── 11_alibi/                  ← get_alibi_slopes, build_alibi_bias, ALiBiAttention
│
├── phase4_architecture/
│   ├── 12_mixture_of_experts/     ← Expert, Router, MoELayer (with load balancing)
│   ├── 13_parallel_transformer/   ← ParallelTransformerBlock vs SequentialTransformerBlock
│   └── 14_swiglu_rmsnorm/         ← RMSNorm, SwiGLU, LLaMABlock, benchmark_norms
│
├── phase5_inference_efficiency/
│   ├── 15_kv_cache/               ← KVCache, CachedAttention, benchmark_cache
│   ├── 16_speculative_decoding/   ← speculative_decode (accept/reject algorithm)
│   ├── 17_quantization/           ← quantize_symmetric, per_channel, QuantizedLinear, quantize_model
│   └── 18_distillation/           ← distillation_loss, hidden_state_loss, DistillationTrainer
│
├── phase6_peft/
│   ├── 19_lora/                   ← LoRALinear, inject_lora, merge/unmerge
│   ├── 20_qlora/                  ← NF4_QUANTIZATION_TABLE, quantize_nf4, NF4Linear
│   └── 21_prefix_tuning/          ← PrefixEncoder, PrefixAttention, PrefixTransformer
│
└── phase7_training_efficiency/
    ├── 22_gradient_checkpointing/ ← CheckpointedTransformerBlock, GPTWithCheckpointing
    ├── 23_mixed_precision/        ← ManualLossScaler, train_amp, BF16TrainingWrapper
    └── 24_lr_schedulers/          ← cosine+warmup, WSD, linear warmup, plot_schedules
```

---

## How to Use Each Project

Every `model.py` has a standalone `if __name__ == "__main__"` block that demonstrates the module. Run any file directly to verify it works:

```bash
cd phase2_attention/06_flash_attention
python model.py
# Output: Max absolute difference (naive vs flash): 1.23e-07 — Outputs match!
```

```bash
cd phase1_foundations/02_gpt_transformer
python train.py --n_layer 4 --n_embd 128 --n_head 4
# Trains in ~2 min on CPU, val loss ~1.8
```

```bash
cd phase6_peft/19_lora
python model.py
# Before LoRA: 3,984,705 trainable / 3,984,705 total
# After  LoRA:     16,384 trainable / 3,984,705 total  (0.41%)
```
