"""Configuration schema for the Transformer model.

A single dataclass keeps every architectural choice (widths, depth, head
counts, positional scheme, norm type, activation) in one explicit, typed
place so a YAML config file in `configs/` maps onto it 1:1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GPTConfig:
    # -- Vocabulary / sequence -------------------------------------------------
    vocab_size: int = 512
    block_size: int = 128          # max context length (a.k.a. n_ctx)

    # -- Width / depth ----------------------------------------------------------
    d_model: int = 256             # embedding / residual-stream dimension
    n_layers: int = 4
    n_heads: int = 4               # number of query heads
    n_kv_heads: int = field(default=0)  # 0 => same as n_heads (plain MHA).
    # Set n_kv_heads < n_heads for Grouped-Query Attention (GQA); n_kv_heads=1
    # is Multi-Query Attention (MQA).
    d_ff: int = 1024               # feedforward hidden dimension

    # -- Positional scheme --------------------------------------------------
    pos_encoding: Literal["learned", "rope"] = "learned"
    rope_theta: float = 10000.0

    # -- Norm / activation ----------------------------------------------------
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    activation: Literal["gelu", "swiglu"] = "gelu"
    norm_eps: float = 1e-5

    # -- Regularization -----------------------------------------------------
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # -- Misc -----------------------------------------------------------------
    bias: bool = True              # whether Linear/Norm layers carry a bias
    tie_embeddings: bool = True    # tie input embedding & output LM head weights

    def __post_init__(self):
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (for grouped-query attention)"
        )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str) -> "GPTConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        # allow a top-level "model:" key, or the fields directly at the root
        raw = raw.get("model", raw)
        return cls(**raw)
