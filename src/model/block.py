"""One Transformer decoder layer: pre-norm attention + pre-norm feedforward,
each wrapped in a residual connection.

Pre-norm (norm -> sublayer -> add residual) rather than post-norm
(sublayer -> add residual -> norm) is used throughout, because pre-norm
keeps a clean, unnormalized residual "stream" running through the whole
network, which is what makes very deep Transformers trainable without
elaborate learning-rate warmup tricks.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.config import GPTConfig
from src.model.feedforward import build_feedforward
from src.model.norm import build_norm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = build_norm(cfg)
        self.ffn = build_feedforward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache=None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        attn_out, present_kv = self.attn(
            self.ln1(x), rope_cache=rope_cache, past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out                 # residual connection 1
        x = x + self.ffn(self.ln2(x))    # residual connection 2
        return x, present_kv
