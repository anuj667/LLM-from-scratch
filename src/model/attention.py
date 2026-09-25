"""Causal self-attention, implemented from the underlying matmuls.

Supports:
  - Plain Multi-Head Attention (MHA): ``n_kv_heads == n_heads``.
  - Grouped-Query Attention (GQA): ``n_kv_heads < n_heads`` -- multiple
    query heads share the same key/value head, cutting the KV cache size
    (this is what most modern production LLMs use).
  - Multi-Query Attention (MQA), the extreme case ``n_kv_heads == 1``.
  - RoPE, if the model config asks for it.
  - An optional KV cache (``past_kv`` in / ``present_kv`` out) for fast
    autoregressive decoding -- see ``src/inference/kv_cache.py``.

The causal mask guarantees position *i* can only attend to positions
``<= i``: this is what makes the model a valid autoregressive language
model rather than a bidirectional encoder. ``tests/test_causal_mask.py``
checks this property directly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import GPTConfig
from src.model.embedding import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # query heads per kv head

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Whether PyTorch's fused, memory-efficient SDPA kernel is available.
        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand (B, n_kv_heads, T, D) -> (B, n_kv_heads * n_rep, T, D) by
        repeating each kv head ``n_rep`` times, so it lines up with the
        (larger) number of query heads for GQA/MQA."""
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, H, n_rep, T, D)
        return x.reshape(B, H * n_rep, T, D)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        x: (B, T, d_model)
        rope_cache: optional (cos, sin) tables from precompute_rope_cache
        past_kv: optional (past_k, past_v), each (B, n_kv_heads, T_past, head_dim),
                 used for fast autoregressive decoding.
        Returns: (out, present_kv) where present_kv is (k, v) if use_cache else None.
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, nH, T, D)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nKV, T, D)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nKV, T, D)

        if rope_cache is not None:
            cos, sin = rope_cache
            # When decoding with a KV cache, the new token's position is
            # offset by how much past context already exists.
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q = apply_rope(q, cos[offset:], sin[offset:])
            k = apply_rope(k, cos[offset:], sin[offset:])

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Expand kv heads up to the number of query heads for GQA/MQA.
        k_rep = self._repeat_kv(k, self.n_rep)
        v_rep = self._repeat_kv(v, self.n_rep)

        T_q = q.shape[2]
        T_kv = k_rep.shape[2]
        offset = T_kv - T_q  # how much cached history precedes this query chunk

        # General causal rule: query position i (0-indexed within this
        # chunk) may attend to key positions <= offset + i. This covers
        # every case uniformly -- a full first pass (offset=0), decoding
        # one new token against a cache (T_q=1, offset=T_kv-1, so the
        # single query trivially sees the whole cache), and even decoding
        # several new tokens at once against a partial cache.
        needs_mask = T_q > 1  # a single query position never needs masking:
        # it always attends to the whole (offset+1)-length key set anyway.

        if self._has_sdpa:
            if not needs_mask:
                attn_mask = None
                is_causal = False
            elif offset == 0:
                attn_mask = None
                is_causal = True
            else:
                causal = torch.ones(T_q, T_kv, dtype=torch.bool, device=x.device).tril(offset)
                attn_mask = torch.zeros(T_q, T_kv, dtype=q.dtype, device=x.device)
                attn_mask.masked_fill_(~causal, float("-inf"))
                is_causal = False
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scale = 1.0 / (self.head_dim ** 0.5)
            att = (q @ k_rep.transpose(-2, -1)) * scale  # (B, nH, T_q, T_kv)
            if needs_mask:
                causal = torch.ones(T_q, T_kv, dtype=torch.bool, device=x.device).tril(offset)
                att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v_rep  # (B, nH, T_q, D)

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(out))
        return out, present_kv
