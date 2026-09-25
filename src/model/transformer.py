"""The full decoder-only Transformer (a "GPT"): stacks embeddings, N
Transformer blocks, a final norm, and an LM head that projects back to
vocabulary logits.

    input token ids
          |
    TokenEmbedding (+ learned positions, or nothing if using RoPE)
          |
    [TransformerBlock] x n_layers   (RoPE, if enabled, applied inside attention)
          |
    final norm
          |
    LM head (Linear d_model -> vocab_size, optionally tied to token_emb.weight)
          |
    logits (B, T, vocab_size)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.block import TransformerBlock
from src.model.config import GPTConfig
from src.model.embedding import TokenEmbedding, precompute_rope_cache
from src.model.norm import build_norm

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = TokenEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = build_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Weight tying: the input embedding matrix and the output
            # projection share the same weight tensor. This roughly halves
            # the embedding-related parameter count and is standard
            # practice (GPT-2, LLaMA, etc.) since embeddings and unembeddings
            # occupy the same semantic space.
            self.lm_head.weight = self.embed.token_emb.weight

        if cfg.pos_encoding == "rope":
            cos, sin = precompute_rope_cache(
                cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual projections: scale down the
        # output projection of each attention/FFN block by 1/sqrt(2*n_layers)
        # so the variance of the residual stream doesn't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight") \
                    or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.embed.pos_emb is not None:
            n -= self.embed.pos_emb.weight.numel()
        return n

    def _rope_cache_for(self, device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.rope_cos is None:
            return None
        return self.rope_cos.to(device), self.rope_sin.to(device)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ):
        """
        token_ids: (B, T) int64
        targets:   (B, T) int64, optional -- if given, computes cross-entropy loss
        past_kv:   optional list of (k, v) per layer, for cached decoding
        Returns: (logits, loss, present_kv)
        """
        x = self.embed(token_ids)
        rope_cache = self._rope_cache_for(token_ids.device)

        present_kv: Optional[KVCache] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_present = block(x, rope_cache=rope_cache, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                present_kv.append(layer_present)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size) -- need all positions for loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # At inference time we only need the logits for the *last*
            # position (the next-token prediction); slicing before the LM
            # head saves a fair bit of compute for long prompts.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, present_kv

    @torch.no_grad()
    def estimate_flops_per_token(self) -> float:
        """Rough 6N approximation (Kaplan et al., 2020) of forward+backward
        FLOPs per token, useful for back-of-envelope compute budgeting."""
        return 6 * self.num_params(non_embedding=True)
