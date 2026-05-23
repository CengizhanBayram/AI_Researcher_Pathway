# Bigram Language Model

## Core Idea

A bigram model defines the probability of a sequence as:

```
P(w_1, w_2, ..., w_T) = ∏ P(w_t | w_{t-1})
```

Each next-token probability depends **only** on the current token.
In neural form, this is just a learned embedding table: `logits = E[x_t]` where `E ∈ R^{V×V}`.

## What You Learn Here

- How character-level tokenization works
- How `cross_entropy` loss connects to perplexity: `loss = -log P(correct token)`
- How `torch.multinomial` samples from a distribution
- Why context length matters — the bigram sees nothing before `x_t`

## Expected Results

Training on tiny shakespeare (~1M chars, char-level):
- Train loss after 3000 steps: ~2.5
- Val loss: ~2.5
- Generated text: character-level gibberish with vaguely English feel

## Next Step

Project 02 (GPT Transformer) uses the same training loop but replaces the
bigram lookup with a full causal attention stack. Watch the val loss drop
to ~1.5 and the text become coherent.

## Key Questions to Answer

1. Why does the bigram model overfit even with a huge val set? (It can't — there are only V² parameters)
2. What is the theoretical lower bound of loss for a perfect bigram model?
3. Why is `cross_entropy` the right loss for language modeling?
