"""Autoregressive generation: repeatedly predict the next token, sample
one, append it, and feed it back in -- using a KV cache so each new token
only costs a single-token forward pass rather than reprocessing the whole
sequence.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.inference.kv_cache import KVCache
from src.inference.sampler import sample_next_token
from src.model.transformer import GPT
from src.tokenizer.bpe import BPETokenizer


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = None,
    greedy: bool = False,
    device: str = "cpu",
) -> str:
    model.eval()
    model.to(device)

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("prompt encoded to zero tokens")

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    cache = KVCache.empty(model.cfg.n_layers)

    # Prime the cache with the full prompt in one forward pass.
    logits, _, present_kv = model(tokens, use_cache=True)
    cache.update(present_kv)

    generated = prompt_ids.copy()
    next_input = tokens[:, -1:]  # placeholder; real next input comes from sampling below
    last_logits = logits[:, -1, :]  # (1, vocab_size)

    for _ in range(max_new_tokens):
        next_id = sample_next_token(
            last_logits, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy
        )  # (1, 1)
        generated.append(next_id.item())

        # Keep the KV cache within the model's trained context length.
        cache.trim_to(model.cfg.block_size - 1)

        logits, _, present_kv = model(next_id, past_kv=cache.as_list(), use_cache=True)
        cache.update(present_kv)
        last_logits = logits[:, -1, :]

    return tokenizer.decode(generated)
