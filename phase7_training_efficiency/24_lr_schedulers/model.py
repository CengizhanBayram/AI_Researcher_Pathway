"""
Learning Rate Schedulers for LLM Training — from scratch.

LLM training is highly sensitive to the learning rate schedule.
The standard recipe: warmup → cosine decay (→ optional constant tail).

Modern variant: WSD (Warmup-Stable-Decay), used in MiniCPM and others.

We implement from scratch:
  1. Linear warmup
  2. Cosine decay
  3. Warmup + cosine combined (GPT-3, LLaMA style)
  4. WSD (Warmup Stable Decay)

References:
  GPT-3 training: cosine + warmup — Brown et al., 2020
  WSD: "MiniCPM" — Hu et al., 2024 — https://arxiv.org/abs/2404.06395
"""

import math
import torch
import torch.optim as optim


def get_cosine_schedule_with_warmup(
    optimizer:    optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
    min_lr_ratio: float = 0.1,
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear warmup for `warmup_steps` steps, then cosine decay to `min_lr_ratio * lr`.

    lr(t) = lr_max * t / warmup_steps                    for t < warmup_steps
    lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1+cos(π*progress))  otherwise

    This is the most common schedule for LLM training (GPT-3, LLaMA).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(
    optimizer:     optim.Optimizer,
    warmup_steps:  int,
    stable_steps:  int,
    decay_steps:   int,
    min_lr_ratio:  float = 0.0,
) -> optim.lr_scheduler.LambdaLR:
    """
    WSD: Warmup → Stable → Decay

    Phase 1 Warmup:  t in [0, warmup)       → linear 0 → lr_max
    Phase 2 Stable:  t in [warmup, stable)  → constant lr_max
    Phase 3 Decay:   t in [stable, end]     → cosine lr_max → lr_min

    Advantage over cosine: you can extend training by just extending the
    stable phase without recomputing the decay schedule. This enables
    continuous training (train, then decay whenever you want to stop).
    """
    total = warmup_steps + stable_steps + decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            decay_progress = min(decay_progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer:    optim.Optimizer,
    warmup_steps: int,
    total_steps:  int,
) -> optim.lr_scheduler.LambdaLR:
    """Simple linear decay (used in BERT fine-tuning)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_schedules(total_steps: int = 10000, warmup: int = 500, save: bool = True):
    """Plot all schedules for visual comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    dummy = torch.nn.Linear(1, 1)
    opt   = optim.AdamW(dummy.parameters(), lr=1.0)  # lr=1 so lambda gives the multiplier

    schedulers = {
        "Cosine + Warmup":  get_cosine_schedule_with_warmup(opt, warmup, total_steps),
        "WSD":              get_wsd_schedule(opt, warmup, total_steps // 2, total_steps // 4),
        "Linear + Warmup":  get_linear_schedule_with_warmup(opt, warmup, total_steps),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, sched in schedulers.items():
        lrs = []
        opt = optim.AdamW(dummy.parameters(), lr=1.0)
        sched = sched.__class__(opt, sched.lr_lambdas[0])  # rebuild with fresh opt
        for step in range(total_steps):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        ax.plot(lrs, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("LR multiplier")
    ax.set_title("LR Schedules")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig("lr_schedules.png", dpi=120)
        print("Saved lr_schedules.png")
    plt.show()


def lr_schedule_values(schedule_fn, n_steps: int) -> list[float]:
    """Get the LR values at each step for inspection."""
    dummy = torch.nn.Linear(1, 1)
    opt   = torch.optim.AdamW(dummy.parameters(), lr=1.0)
    sched = schedule_fn(opt)
    vals  = []
    for _ in range(n_steps):
        vals.append(opt.param_groups[0]["lr"])
        sched.step()
    return vals


if __name__ == "__main__":
    dummy = torch.nn.Linear(1, 1)
    opt   = torch.optim.AdamW(dummy.parameters(), lr=3e-4)

    total  = 10000
    warmup = 500

    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=warmup, total_steps=total)

    for step in [0, 100, 500, 1000, 5000, 9999]:
        print(f"Step {step:5d}: lr = {opt.param_groups[0]['lr']:.6f}")
        # Fast-forward
        for _ in range(step if step == 0 else 1):
            sched.step()

    plot_schedules(total_steps=10000, warmup=500)
