"""The pretraining loop itself.

Each step:
  1. Sample a random batch of (x, y) context windows straight off the
     memory-mapped token file.
  2. Forward pass under automatic mixed precision (AMP) when running on
     CUDA, for roughly 2-3x throughput with no accuracy loss.
  3. Backward pass, gradient clipping (caps the global gradient norm --
     the single most effective trick for keeping Transformer pretraining
     numerically stable), optimizer step, LR schedule update.
  4. Periodic validation-loss evaluation and checkpointing.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.data.dataloader import get_random_batch
from src.model.transformer import GPT
from src.training.lr_scheduler import CosineWarmupScheduler
from src.training.optimizer import build_optimizer
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import MetricsLogger


@dataclass
class TrainingConfig:
    train_bin: str = "data/processed/train.bin"
    val_bin: str = "data/processed/val.bin"
    out_dir: str = "checkpoints"

    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 500

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = "bfloat16"  # "float32" | "float16" | "bfloat16"
    seed: int = 1337

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        raw = raw.get("training", raw)
        return cls(**raw)


class Trainer:
    def __init__(self, model: GPT, cfg: TrainingConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.optimizer = build_optimizer(model, lr=cfg.max_lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineWarmupScheduler(
            max_lr=cfg.max_lr, min_lr=cfg.min_lr,
            warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps,
        )

        self.amp_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.dtype]
        self.use_amp = self.device.type == "cuda" and cfg.dtype != "float32"
        # GradScaler is only needed for float16 (bfloat16 has enough
        # dynamic range that it doesn't underflow the way float16 does).
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and cfg.dtype == "float16")

        self.logger = MetricsLogger(os.path.join(cfg.out_dir, "train_log.jsonl"))
        torch.manual_seed(cfg.seed)

        os.makedirs(cfg.out_dir, exist_ok=True)

    def _get_batch(self, split: str):
        bin_path = self.cfg.train_bin if split == "train" else self.cfg.val_bin
        return get_random_batch(
            bin_path,
            block_size=self.model.cfg.block_size,
            vocab_size=self.model.cfg.vocab_size,
            batch_size=self.cfg.batch_size,
            device=self.device.type,
        )

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.cfg.eval_iters)
            for i in range(self.cfg.eval_iters):
                x, y = self._get_batch(split)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss, _ = self.model(x, targets=y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def load_resume_state(self, checkpoint_path: str) -> int:
        """Load model + optimizer state from a checkpoint and return the
        step to resume from. Call this before ``train()`` when continuing
        an interrupted or extended run, rather than starting over."""
        from src.utils.checkpoint import load_checkpoint

        ckpt = load_checkpoint(checkpoint_path, map_location=str(self.device))
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("warning: checkpoint has no optimizer state (inference-only "
                  "checkpoint?); resuming with a freshly initialized optimizer.")
        start_step = ckpt.get("step", 0)
        print(f"resumed from {checkpoint_path} at step {start_step}")
        return start_step

    def train(self, start_step: int = 0):
        self.model.train()
        t0 = time.time()
        running_loss = 0.0

        if start_step >= self.cfg.max_steps:
            print(f"start_step ({start_step}) >= max_steps ({self.cfg.max_steps}); "
                  f"nothing to do. Increase max_steps in your config to keep training.")
            return

        for step in range(start_step, self.cfg.max_steps):
            lr = self.scheduler.step(self.optimizer, step)

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro_step in range(self.cfg.grad_accum_steps):
                x, y = self._get_batch("train")
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss, _ = self.model(x, targets=y)
                    loss = loss / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += step_loss

            if step % self.cfg.log_interval == 0:
                dt = time.time() - t0
                avg_loss = running_loss / max(1, self.cfg.log_interval if step > 0 else 1)
                tokens_per_sec = (
                    self.cfg.batch_size * self.model.cfg.block_size * self.cfg.grad_accum_steps
                    * (self.cfg.log_interval if step > 0 else 1) / max(dt, 1e-9)
                )
                print(f"step {step:6d} | loss {step_loss:.4f} | lr {lr:.2e} "
                      f"| {tokens_per_sec:,.0f} tok/s")
                self.logger.log({"step": step, "loss": step_loss, "lr": lr})
                running_loss = 0.0
                t0 = time.time()

            if step > 0 and step % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"  [eval] step {step}: train_loss={losses['train']:.4f} "
                      f"val_loss={losses['val']:.4f}")
                self.logger.log({"step": step, "eval_train_loss": losses["train"],
                                  "eval_val_loss": losses["val"]})

            if step > 0 and step % self.cfg.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.cfg.out_dir, f"step_{step}.pt")
                save_checkpoint(ckpt_path, self.model, self.optimizer, step)
                print(f"  saved checkpoint -> {ckpt_path}")

        # Always save a final checkpoint.
        final_path = os.path.join(self.cfg.out_dir, f"step_{self.cfg.max_steps}.pt")
        save_checkpoint(final_path, self.model, self.optimizer, self.cfg.max_steps)
        print(f"training complete. final checkpoint -> {final_path}")
