"""
Train the GPT-style transformer.

Usage:
    python train.py                    # tiny shakespeare, default config
    python train.py --n_layer 4 --n_embd 256 --n_head 4   # smaller model
    python train.py --text my_corpus.txt
"""

import argparse
import urllib.request
from pathlib import Path

import torch
from model import GPT, GPTConfig

# ── hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE  = 64
BLOCK_SIZE  = 256
MAX_ITERS   = 5000
EVAL_ITERS  = 200
EVAL_EVERY  = 500
LR          = 3e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────


def load_text(path: str | None) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading tiny shakespeare ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return (
        lambda s: [stoi[c] for c in s],
        lambda l: "".join(itos[i] for i in l),
        len(chars),
    )


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i     : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",    default=None)
    parser.add_argument("--n_layer", type=int,   default=6)
    parser.add_argument("--n_embd",  type=int,   default=384)
    parser.add_argument("--n_head",  type=int,   default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    text = load_text(args.text)
    encode, decode, vocab_size = build_vocab(text)
    print(f"Vocab size: {vocab_size}  |  Corpus: {len(text):,} chars")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(
        vocab_size  = vocab_size,
        block_size  = BLOCK_SIZE,
        n_embd      = args.n_embd,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        dropout     = args.dropout,
    )
    model = GPT(config).to(DEVICE)

    # Cosine LR with linear warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    print(f"\nTraining on {DEVICE} | {MAX_ITERS} iters\n")
    for step in range(MAX_ITERS):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("\n── Sample generation (temperature=0.8, top_k=40) ────────────")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    tokens = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()
    print(decode(tokens))

    torch.save(model.state_dict(), "gpt_weights.pt")
    print("\nWeights saved to gpt_weights.pt")


if __name__ == "__main__":
    main()
