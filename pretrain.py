#!/usr/bin/env python3
"""Launch a pretraining run.

Usage:
    python pretrain.py --config configs/gpt_nano.yaml
"""
from __future__ import annotations

import argparse

import yaml

from src.model.config import GPTConfig
from src.model.transformer import GPT
from src.training.trainer import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Pretrain a GPT-style model from scratch")
    parser.add_argument("--config", type=str, required=True,
                         help="YAML config with `model:` and `training:` sections")
    parser.add_argument("--resume", type=str, default=None,
                         help="path to a checkpoint (e.g. checkpoints/step_2000.pt) to "
                              "resume training from, instead of starting over from step 0. "
                              "Raise max_steps in your config first so there are more steps "
                              "left to run.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    model_cfg = GPTConfig(**raw["model"])
    train_cfg = TrainingConfig(**raw.get("training", {}))

    print("Model config:", model_cfg)
    model = GPT(model_cfg)
    print(f"Model has {model.num_params():,} non-embedding parameters "
          f"({model.num_params(non_embedding=False):,} total)")

    trainer = Trainer(model, train_cfg)

    start_step = 0
    if args.resume:
        start_step = trainer.load_resume_state(args.resume)

    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()
