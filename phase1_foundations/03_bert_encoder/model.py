"""
BERT-style Bidirectional Encoder — from scratch.

Architecture:
  Token Embedding + Positional Embedding + Segment Embedding
  → N × TransformerBlock (full bidirectional attention — no causal mask)
  → MLM Head  (predict masked tokens)
  → NSP Head  (predict if sentence B follows sentence A)

Reference: "BERT: Pre-training of Deep Bidirectional Transformers"
           Devlin et al., 2018 — https://arxiv.org/abs/1810.04805
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BERTConfig:
    vocab_size:   int   = 30522  # standard BERT wordpiece vocab
    max_seq_len:  int   = 512
    n_embd:       int   = 256    # BERT-base uses 768; we use smaller for experiments
    n_head:       int   = 4
    n_layer:      int   = 4
    ffn_mult:     int   = 4      # FFN inner dim = ffn_mult * n_embd
    dropout:      float = 0.1
    n_segments:   int   = 2      # sentence A / sentence B


MASK_TOKEN_ID = 103   # [MASK] in BERT wordpiece
CLS_TOKEN_ID  = 101
SEP_TOKEN_ID  = 102
PAD_TOKEN_ID  = 0


class BidirectionalSelfAttention(nn.Module):
    """
    Full (non-causal) self-attention. Every position can attend to every other position.
    The only difference from CausalSelfAttention: NO causal mask.
    """

    def __init__(self, config: BERTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head   = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        if padding_mask is not None:
            # padding_mask: (B, T)  — True where token is padding
            att = att.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v                                               # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class BERTBlock(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ffn  = nn.Sequential(
            nn.Linear(config.n_embd, config.ffn_mult * config.n_embd),
            nn.GELU(),
            nn.Linear(config.ffn_mult * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class MLMHead(nn.Module):
    """Masked Language Model head: predict identity of [MASK] tokens."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.ln    = nn.LayerNorm(config.n_embd)
        self.proj  = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(F.gelu(self.dense(hidden))))


class NSPHead(nn.Module):
    """Next Sentence Prediction head: binary classification on the [CLS] token."""

    def __init__(self, config: BERTConfig):
        super().__init__()
        self.classifier = nn.Linear(config.n_embd, 2)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls_hidden)


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size,  config.n_embd, padding_idx=PAD_TOKEN_ID)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.seg_emb = nn.Embedding(config.n_segments,  config.n_embd)
        self.emb_drop = nn.Dropout(config.dropout)
        self.emb_ln   = nn.LayerNorm(config.n_embd)

        self.blocks = nn.ModuleList([BERTBlock(config) for _ in range(config.n_layer)])

        self.mlm_head = MLMHead(config)
        self.nsp_head = NSPHead(config)

        self.apply(self._init_weights)
        print(f"BERT initialized | params: {self.num_params():,}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(
        self,
        input_ids:      torch.Tensor,
        segment_ids:    torch.Tensor | None = None,
        padding_mask:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        x = self.tok_emb(input_ids) + self.pos_emb(positions) + self.seg_emb(segment_ids)
        x = self.emb_ln(self.emb_drop(x))

        for block in self.blocks:
            x = block(x, padding_mask)

        return x  # (B, T, n_embd)

    def forward(
        self,
        input_ids:    torch.Tensor,
        segment_ids:  torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        mlm_labels:   torch.Tensor | None = None,
        nsp_labels:   torch.Tensor | None = None,
    ):
        hidden = self.encode(input_ids, segment_ids, padding_mask)  # (B, T, n_embd)
        cls_hidden = hidden[:, 0, :]                                  # [CLS] token

        mlm_logits = self.mlm_head(hidden)  # (B, T, vocab_size)
        nsp_logits = self.nsp_head(cls_hidden)  # (B, 2)

        total_loss = None
        if mlm_labels is not None:
            # Only compute loss on masked positions (label = -100 for unmasked)
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = mlm_loss

        if nsp_labels is not None:
            nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
            total_loss = (total_loss + nsp_loss) if total_loss is not None else nsp_loss

        return {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits, "loss": total_loss}


def apply_mlm_masking(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    mask_token_id: int = MASK_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    BERT masking strategy:
      80% of selected tokens → [MASK]
      10% → random token
      10% → unchanged
    Labels are -100 (ignore) for unmasked positions.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mask_prob)

    # Never mask special tokens
    special = (input_ids == CLS_TOKEN_ID) | (input_ids == SEP_TOKEN_ID) | (input_ids == PAD_TOKEN_ID)
    probability_matrix[special] = 0.0

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% → [MASK]
    replace_with_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[replace_with_mask] = mask_token_id

    # 10% → random token (of the remaining 20%)
    replace_with_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~replace_with_mask
    random_tokens = torch.randint(vocab_size, input_ids.shape, dtype=input_ids.dtype)
    input_ids[replace_with_random] = random_tokens[replace_with_random]

    # 10% → unchanged (already handled by not replacing)
    return input_ids, labels
