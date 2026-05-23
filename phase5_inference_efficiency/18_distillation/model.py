"""
Knowledge Distillation — from scratch.

Transfer knowledge from a large "teacher" model to a smaller "student" model.
The student is trained to mimic the teacher's soft probability distributions
(not just the hard labels), learning more information per training step.

Loss:
  L = α * L_CE(student_logits, hard_labels)
    + (1-α) * T² * KL(softmax(student_logits/T) || softmax(teacher_logits/T))

where T = temperature (softens distributions, revealing more information about
inter-class relationships), and α balances hard vs soft targets.

Reference: "Distilling the Knowledge in a Neural Network"
           Hinton et al., 2015 — https://arxiv.org/abs/1503.02531

LLM-specific: "DistilBERT" (Sanh et al., 2019), TinyLLaMA, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    temperature: float = 4.0    # higher T → softer distributions → more info
    alpha:       float = 0.5    # weight of hard label loss (1-α = soft label weight)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined distillation loss.

    student_logits: (B*T, vocab_size) — raw student output
    teacher_logits: (B*T, vocab_size) — raw teacher output (no_grad)
    labels:         (B*T,)            — ground truth token ids

    Returns: (total_loss, hard_loss, soft_loss)
    """
    T = config.temperature
    α = config.alpha

    # Hard label loss: standard cross entropy on ground truth
    hard_loss = F.cross_entropy(student_logits, labels, ignore_index=-100)

    # Soft label loss: KL divergence between student and teacher distributions
    # Scale by T² to compensate for the smaller gradients from softened distributions
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits  / T, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

    total_loss = α * hard_loss + (1 - α) * soft_loss
    return total_loss, hard_loss, soft_loss


def hidden_state_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection: nn.Linear | None = None,
) -> torch.Tensor:
    """
    Feature-level distillation: match intermediate hidden states.
    Used in DistilBERT and TinyBERT.

    If student and teacher have different dimensions, use a learned projection.
    """
    if projection is not None:
        student_hidden = projection(student_hidden)
    return F.mse_loss(student_hidden, teacher_hidden.detach())


def attention_distillation_loss(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
) -> torch.Tensor:
    """
    Attention transfer: student attention maps match teacher attention maps.
    Used in TinyBERT.
    student_attn: (B, n_head_s, T, T)
    teacher_attn: (B, n_head_t, T, T)

    If head counts differ, average across heads first.
    """
    s = student_attn.mean(dim=1)  # (B, T, T)
    t = teacher_attn.mean(dim=1)  # (B, T, T)
    return F.mse_loss(s, t.detach())


class DistillationTrainer:
    """
    Manages distillation training loop.
    Teacher is frozen; only student is updated.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.config  = config
        self.device  = device
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

        # Freeze teacher completely
        for p in self.teacher.parameters():
            p.requires_grad = False

    def step(self, input_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        input_ids = input_ids.to(self.device)
        targets   = targets.to(self.device)

        # Teacher forward (no gradient needed)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(input_ids)

        # Student forward
        student_logits, _ = self.student(input_ids)

        B, T, C = student_logits.shape
        total_loss, hard_loss, soft_loss = distillation_loss(
            student_logits.view(B * T, C),
            teacher_logits.view(B * T, C),
            targets.view(B * T),
            self.config,
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "hard_loss":  hard_loss.item(),
            "soft_loss":  soft_loss.item(),
        }


def demo_distillation():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../phase1_foundations/02_gpt_transformer"))
    from model import GPT, GPTConfig

    teacher_cfg = GPTConfig(vocab_size=65, block_size=64, n_embd=256, n_head=8, n_layer=6)
    student_cfg = GPTConfig(vocab_size=65, block_size=64, n_embd=128, n_head=4, n_layer=3)

    teacher = GPT(teacher_cfg)
    student = GPT(student_cfg)

    print(f"Teacher params: {teacher.num_params():,}")
    print(f"Student params: {student.num_params():,}")
    print(f"Compression:    {teacher.num_params() / student.num_params():.1f}×")

    trainer = DistillationTrainer(teacher, student, DistillationConfig())

    x = torch.randint(0, 65, (4, 64))
    y = torch.randint(0, 65, (4, 64))
    metrics = trainer.step(x, y)
    print(f"\nStep metrics: {metrics}")


if __name__ == "__main__":
    demo_distillation()
