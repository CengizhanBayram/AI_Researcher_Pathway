# GPT-style Autoregressive Transformer

## Papers
- "Attention Is All You Need" — Vaswani et al., 2017
- "Language Models are Unsupervised Multitask Learners" (GPT-2) — Radford et al., 2019

## Key Equations

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```
The 1/sqrt(d_k) scaling prevents dot products from growing large
and pushing softmax into saturated (near-zero gradient) regions.

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
```
Each head can attend to different aspects of the sequence.

### Causal Mask
```
mask[i][j] = 0 if j > i else 1
att = softmax((QK^T / sqrt(d_k)) + log(mask))
```
Setting masked positions to -inf before softmax → probability = 0.

### Pre-Norm vs Post-Norm
Original paper: x = LayerNorm(x + SubLayer(x))  [post-norm]
GPT-2 onward:   x = x + SubLayer(LayerNorm(x))  [pre-norm]

Pre-norm is more stable because gradients flow cleanly through
the residual path without going through LayerNorm.

### Weight Tying
Embedding matrix E ∈ R^{V×d} is shared with the output projection W ∈ R^{d×V}.
- Reduces parameters by V×d
- Works because "words similar in meaning should be close in embedding space AND
  have similar output logit patterns"

## Design Decisions to Understand

1. **Why no bias in attention projections?**
   Bias terms add almost nothing but hurt weight sharing / reg. Modern practice skips them.

2. **Why GELU over ReLU in FFN?**
   GELU is smooth at zero, empirically better for language tasks.

3. **Why gradient clipping at 1.0?**
   Exploding gradients in deep transformers. Clip norm, not values.

4. **Why AdamW over Adam?**
   AdamW decouples weight decay from the adaptive step, which matters for LLMs.

## Expected Results

Tiny shakespeare, default config (~10M params):
- 5000 iters, ~5 min on CPU / 1 min on GPU
- Train loss: ~1.2
- Val loss: ~1.5
- Output: recognizable Shakespearean prose

Compare to bigram val loss of ~2.5 — that's the gain from attention.
