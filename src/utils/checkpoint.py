"""Save/load model + optimizer + training-step state.

A checkpoint bundles everything needed to either resume training exactly
where it left off, or to load just the model weights for inference:

    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,   # omitted when loading for inference only
        "step": ...,
        "model_config": ...,           # the GPTConfig, as a dict, so a checkpoint
                                        # is self-describing and doesn't require
                                        # the caller to already know the architecture
    }
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch


def save_checkpoint(path: str, model, optimizer=None, step: int = 0) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "model_config": dataclasses.asdict(model.cfg),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> dict:
    return torch.load(path, map_location=map_location or "cpu")


def load_model_from_checkpoint(path: str, map_location: Optional[str] = None):
    """Reconstruct a GPT model (architecture + weights) purely from a
    checkpoint file, using the config that was saved alongside the
    weights. This is what ``generate.py`` uses, so you never have to
    remember (or hand-specify) the exact architecture a checkpoint was
    trained with."""
    from src.model.config import GPTConfig
    from src.model.transformer import GPT

    ckpt = load_checkpoint(path, map_location=map_location)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
