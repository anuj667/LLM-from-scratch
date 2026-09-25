"""Learning-rate schedule: linear warmup, then cosine decay to a floor.

This is the schedule used by GPT-2/GPT-3 and nearly every LLM pretraining
run since: warming up avoids the large, unstable early-training updates
that come from Adam's second-moment estimate being poorly calibrated at
step 0, and cosine decay (rather than a step schedule) has empirically
proven to give the best final loss for a fixed training budget.
"""
from __future__ import annotations

import math


class CosineWarmupScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        assert min_lr <= max_lr
        assert warmup_steps <= max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup from ~0 to max_lr.
            return self.max_lr * (step + 1) / max(1, self.warmup_steps)
        if step >= self.max_steps:
            return self.min_lr
        # Cosine decay from max_lr down to min_lr over the remaining steps.
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self, optimizer, step: int) -> float:
        """Compute the LR for ``step`` and apply it to every param group."""
        lr = self.get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
