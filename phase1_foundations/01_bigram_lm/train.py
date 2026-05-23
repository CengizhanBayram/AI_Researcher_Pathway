"""
Train the Bigram LM on a small text corpus (tiny shakespeare by default).

Usage:
    python train.py
    python train.py --text my_corpus.txt
"""

import argparse
import urllib.request
from pathlib import Path

import torch
from model import BigramLM

# ── hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 32
BLOCK_SIZE   = 8       # context length (bigram only uses last token, but we batch T steps)
MAX_ITERS    = 3000
EVAL_ITERS   = 200
EVAL_EVERY   = 300
LR           = 1e-2
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────


def load_text(path: str | None) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading tiny shakespeare from {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    return encode, decode, len(chars)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i     : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters):
    model.eval()
    results = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    text = load_text(args.text)
    encode, decode, vocab_size = build_vocab(text)
    print(f"Vocab size: {vocab_size}  |  Text length: {len(text):,} chars")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = BigramLM(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"Training on {DEVICE} for {MAX_ITERS} iterations ...\n")
    for step in range(MAX_ITERS):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS)
            print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n── Sample generation ────────────────────────────────")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(tokens))


if __name__ == "__main__":
    main()
