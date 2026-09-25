"""AdamW optimizer setup, with the standard weight-decay exclusion rule:
decay all 2D+ parameters (matrix multiplies -- embeddings, linear weights),
but never decay 1D parameters (biases, LayerNorm/RMSNorm scale/shift).

Decaying norm and bias parameters has little regularizing benefit and can
actively hurt training, so nearly every modern LLM training recipe
(GPT-2, GPT-3, LLaMA, ...) splits parameters into two groups like this.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Tied weights (e.g. lm_head sharing token_emb) show up twice in
        # named_parameters(); make sure we only add each tensor once.
        if id(param) in seen:
            continue
        seen.add(id(param))

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"optimizer: {len(decay_params)} decayed tensors ({n_decay:,} params), "
          f"{len(no_decay_params)} non-decayed tensors ({n_no_decay:,} params)")

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Use the fused AdamW kernel when running on CUDA, since it's a
    # meaningful speedup and is safe to opt into if available.
    use_fused = torch.cuda.is_available()
    extra = {"fused": True} if use_fused else {}
    try:
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, **extra)
    except TypeError:
        # Older PyTorch versions without the `fused` kwarg.
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
