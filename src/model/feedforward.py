"""Position-wise feedforward ("MLP") blocks.

Two variants, selected by ``GPTConfig.activation``:

- ``gelu``: the classic GPT-2-style two-layer MLP with a GeLU nonlinearity
  in between: ``Linear(d_model -> d_ff) -> GeLU -> Linear(d_ff -> d_model)``.
- ``swiglu``: the gated variant used by LLaMA/PaLM/Mistral. Instead of one
  up-projection, it computes two ("gate" and "up"), applies SiLU to the
  gate, multiplies them elementwise, then projects back down. Empirically
  this consistently outperforms a plain GeLU MLP at matched parameter
  count, which is why nearly every modern open LLM uses it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import GPTConfig


class GELUMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc_in(x))
        x = self.fc_out(x)
        return self.dropout(x)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # SwiGLU has three weight matrices instead of an MLP's two; to keep
        # parameter count roughly comparable to a GeLU MLP of hidden size
        # `d_ff`, the conventional trick (LLaMA) is to shrink the hidden
        # dim by ~2/3. We keep it simple and configurable-enough here by
        # just using d_ff directly for the gate/up projections.
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        return self.dropout(x)


def build_feedforward(cfg: GPTConfig) -> nn.Module:
    if cfg.activation == "swiglu":
        return SwiGLUMLP(cfg)
    return GELUMLP(cfg)
