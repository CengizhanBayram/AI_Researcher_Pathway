"""
Pre-train BERT with Masked Language Modeling (MLM) on a small text corpus.

NSP (Next Sentence Prediction) is simplified here — we use adjacent sentences
and random sentence pairs from the corpus.

Usage:
    python train.py
    python train.py --text my_corpus.txt
"""

import argparse
import random
import urllib.request
from pathlib import Path

import torch
from model import BERT, BERTConfig, apply_mlm_masking, CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID

BATCH_SIZE = 16
MAX_SEQ_LEN = 128
MAX_ITERS = 3000
EVAL_EVERY = 300
EVAL_ITERS = 50
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_text(path: str | None) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading tiny shakespeare ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def build_char_vocab(text: str):
    chars = sorted(set(text))
    # Reserve 0=PAD, 101=CLS, 102=SEP, 103=MASK
    offset = 104
    stoi = {c: i + offset for i, c in enumerate(chars)}
    itos = {v: k for k, v in stoi.items()}
    vocab_size = offset + len(chars)
    return stoi, itos, vocab_size


def tokenize_sentences(text: str, stoi: dict, max_len: int = 64) -> list[list[int]]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        tokens = [stoi.get(c, PAD_TOKEN_ID) for c in line]
        if len(tokens) > 0:
            sentences.append(tokens[:max_len])
    return sentences


def make_nsp_pair(sentences: list[list[int]], max_seq_len: int):
    idx = random.randint(0, len(sentences) - 2)
    sent_a = sentences[idx]

    is_next = random.random() > 0.5
    sent_b = sentences[idx + 1] if is_next else sentences[random.randint(0, len(sentences) - 1)]
    nsp_label = 0 if is_next else 1  # 0 = IsNext, 1 = NotNext

    # Build input: [CLS] sent_a [SEP] sent_b [SEP]
    tokens = [CLS_TOKEN_ID] + sent_a + [SEP_TOKEN_ID] + sent_b + [SEP_TOKEN_ID]
    segments = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)

    # Pad or truncate
    tokens   = tokens[:max_seq_len]
    segments = segments[:max_seq_len]
    padding_len = max_seq_len - len(tokens)
    tokens   += [PAD_TOKEN_ID] * padding_len
    segments += [0] * padding_len

    padding_mask = [t == PAD_TOKEN_ID for t in tokens]
    return tokens, segments, padding_mask, nsp_label


def make_batch(sentences: list[list[int]], batch_size: int, max_seq_len: int, vocab_size: int):
    all_input, all_segments, all_masks, all_nsp = [], [], [], []
    for _ in range(batch_size):
        t, s, m, n = make_nsp_pair(sentences, max_seq_len)
        all_input.append(t)
        all_segments.append(s)
        all_masks.append(m)
        all_nsp.append(n)

    input_ids    = torch.tensor(all_input,    dtype=torch.long)
    segment_ids  = torch.tensor(all_segments, dtype=torch.long)
    padding_mask = torch.tensor(all_masks,    dtype=torch.bool)
    nsp_labels   = torch.tensor(all_nsp,      dtype=torch.long)

    input_ids, mlm_labels = apply_mlm_masking(input_ids, vocab_size)

    return (
        input_ids.to(DEVICE),
        segment_ids.to(DEVICE),
        padding_mask.to(DEVICE),
        mlm_labels.to(DEVICE),
        nsp_labels.to(DEVICE),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    text = load_text(args.text)
    stoi, itos, vocab_size = build_char_vocab(text)
    sentences = tokenize_sentences(text, stoi, max_len=MAX_SEQ_LEN // 2 - 2)
    print(f"Vocab size: {vocab_size}  |  Sentences: {len(sentences):,}")

    config = BERTConfig(
        vocab_size  = vocab_size,
        max_seq_len = MAX_SEQ_LEN,
        n_embd      = 128,
        n_head      = 4,
        n_layer     = 4,
        dropout     = 0.1,
    )
    model = BERT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"\nTraining on {DEVICE} | {MAX_ITERS} iters\n")
    for step in range(MAX_ITERS):
        input_ids, segment_ids, padding_mask, mlm_labels, nsp_labels = make_batch(
            sentences, BATCH_SIZE, MAX_SEQ_LEN, vocab_size
        )
        out = model(input_ids, segment_ids, padding_mask, mlm_labels, nsp_labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    print("\nTraining complete.")
    torch.save(model.state_dict(), "bert_weights.pt")
    print("Weights saved to bert_weights.pt")


if __name__ == "__main__":
    main()
