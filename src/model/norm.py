"""Normalization layers.

Both are implemented explicitly (rather than only relying on
``torch.nn.LayerNorm``) so the difference between the two is visible:
LayerNorm centers *and* rescales activations using a mean and a variance;
RMSNorm (used by LLaMA, Mistral, etc.) skips the mean-centering step and
just rescales by the root-mean-square, which is cheaper and empirically
works just as well for Transformers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import GPTConfig


class LayerNorm(nn.Module):
    """Standard LayerNorm with an optional bias (PyTorch's always has one)."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (no mean-centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability regardless of the
        # model's working dtype (important under fp16/bf16 autocast).
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(dtype)


def build_norm(cfg: GPTConfig, dim: int | None = None) -> nn.Module:
    dim = dim if dim is not None else cfg.d_model
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(dim, eps=cfg.norm_eps)
    return LayerNorm(dim, bias=cfg.bias, eps=cfg.norm_eps)
