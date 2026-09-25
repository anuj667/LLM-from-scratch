"""Decoding strategies for turning next-token logits into a chosen token id.

- **Greedy**: always take the single highest-probability token. Deterministic,
  but prone to repetitive/degenerate text over long generations.
- **Temperature**: rescale logits by ``1/T`` before softmax. ``T < 1`` sharpens
  the distribution (more confident, less diverse); ``T > 1`` flattens it.
- **Top-k**: zero out every token outside the ``k`` highest-probability
  tokens, then sample from what's left. Caps the "risk" of sampling a
  wildly unlikely token regardless of how flat the distribution is.
- **Top-p / nucleus**: instead of a fixed count, keep the smallest set of
  top tokens whose cumulative probability exceeds ``p``. Adapts to the
  shape of the distribution -- fewer candidates when the model is
  confident, more when it's uncertain -- which is why it tends to produce
  more natural text than top-k at a fixed setting.

All strategies can be combined (e.g. temperature + top-k + top-p together,
applied in that order), matching how most production samplers work.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """
    logits: (B, vocab_size) -- logits for the *next* token only.
    Returns: (B, 1) sampled token ids.
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy=True for argmax decoding)")
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        # Remove tokens once the cumulative probability *exceeds* top_p,
        # but always keep at least the single most likely token.
        remove_mask = cumulative > top_p
        remove_mask[:, 1:] = remove_mask[:, :-1].clone()
        remove_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
