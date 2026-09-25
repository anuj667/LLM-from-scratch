"""Key-Value cache for fast autoregressive decoding.

Without a KV cache, generating token T+1 from a prompt of length T requires
re-running the *entire* forward pass over all T+1 tokens -- quadratic total
work across a generation of length N. With a KV cache, previously computed
keys/values for tokens 0..T are stored and reused, so generating each new
token only requires a forward pass over that *one* new token attending to
the cached keys/values -- linear total work.

This module is a thin, explicit wrapper around the ``present_kv`` lists
already produced by ``GPT.forward(..., use_cache=True)`` (see
``src/model/attention.py`` and ``src/model/transformer.py``); it exists so
callers (``generator.py``) have a small, named object to hold and pass
around rather than juggling raw nested lists/tuples.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

LayerKV = Tuple[torch.Tensor, torch.Tensor]  # (k, v), each (B, n_kv_heads, T, head_dim)


class KVCache:
    """Holds one (k, v) tensor pair per Transformer layer."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._layers: List[Optional[LayerKV]] = [None] * n_layers

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        return cls(n_layers)

    def as_list(self) -> List[Optional[LayerKV]]:
        """Return the raw list-of-tuples format ``GPT.forward`` expects."""
        return self._layers

    def update(self, present_kv: List[LayerKV]) -> None:
        """Replace the cache contents with the (extended) present_kv
        returned by the most recent forward pass."""
        assert len(present_kv) == self.n_layers
        self._layers = list(present_kv)

    @property
    def seq_len(self) -> int:
        """How many tokens' worth of keys/values are currently cached."""
        if self._layers[0] is None:
            return 0
        return self._layers[0][0].shape[2]

    def trim_to(self, max_len: int) -> None:
        """Drop the oldest cached positions, keeping only the most recent
        ``max_len`` -- useful for a sliding-window context during very
        long generations that would otherwise exceed ``block_size``."""
        if self.seq_len <= max_len:
            return
        self._layers = [
            (k[:, :, -max_len:, :], v[:, :, -max_len:, :]) for (k, v) in self._layers
        ]
