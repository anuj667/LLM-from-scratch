"""Token embeddings and positional encodings.

Two positional schemes are implemented, selected via ``GPTConfig.pos_encoding``:

- ``learned``: a standard learned absolute-position embedding table added to
  the token embeddings (GPT-1/GPT-2 style). Simple, but doesn't extrapolate
  well beyond the trained context length.
- ``rope``: Rotary Positional Embeddings. Instead of adding a positional
  vector, RoPE *rotates* pairs of dimensions within each attention head's
  query/key vectors by an angle proportional to position, which bakes
  relative-position information directly into the dot product used by
  attention. RoPE is applied inside the attention module (see
  ``attention.py``); this file only provides ``TokenEmbedding`` (used with
  the "learned" scheme, positions are added here) and the
  ``precompute_rope_cache`` / ``apply_rope`` helpers used by attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import GPTConfig


class TokenEmbedding(nn.Module):
    """Token embedding table, optionally with a learned position table.

    When ``pos_encoding == "rope"``, this module returns *only* the token
    embeddings (RoPE is applied later, inside attention, directly to
    queries/keys -- it is not something you add to the input).
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        else:
            self.pos_emb = None
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids``: (batch, seq_len) int64 -> (batch, seq_len, d_model)."""
        B, T = token_ids.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        x = self.token_emb(token_ids)  # (B, T, d_model)
        if self.pos_emb is not None:
            positions = torch.arange(T, device=token_ids.device)
            x = x + self.pos_emb(positions)[None, :, :]  # broadcast over batch
        return self.dropout(x)


def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0,
                           device=None, dtype=torch.float32):
    """Precompute the cos/sin tables used by RoPE for every position up to
    ``max_seq_len``.

    RoPE treats each head's ``head_dim``-length vector as ``head_dim // 2``
    complex-plane pairs, and rotates pair ``i`` by an angle that grows
    linearly with sequence position but shrinks geometrically across pair
    index (so early dimensions rotate fast / encode fine-grained relative
    position, and late dimensions rotate slowly / encode coarse position).
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    cos = angles.cos().to(dtype=dtype, device=device)
    sin = angles.sin().to(dtype=dtype, device=device)
    return cos, sin  # each (max_seq_len, head_dim // 2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a query or key tensor.

    ``x``: (batch, n_heads, seq_len, head_dim)
    ``cos``, ``sin``: (seq_len, head_dim // 2), already sliced to the
    current sequence length by the caller.
    """
    B, H, T, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2:]  # split into two halves
    cos = cos[None, None, :T, :]
    sin = sin[None, None, :T, :]
    # Standard RoPE rotation: treat (x1, x2) as the real/imag parts of a
    # complex number and multiply by e^{i*angle}.
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)
    return rotated.to(x.dtype)
