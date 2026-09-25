"""Verify future tokens cannot leak into earlier attention steps.

The test strategy: run the model on a sequence, then change only the
*last* token and re-run. If the model is properly causal, every logit at
every position *except the last* must be bit-for-bit identical between the
two runs, because a correctly-masked position i can never see position
T-1 > i. If any earlier logit changes, information has leaked backward
through a broken (or missing) causal mask.

This is checked for every combination of positional scheme (learned/RoPE)
and attention style (plain MHA / GQA), since a masking bug could plausibly
hide in either the SDPA fast path or the manual-softmax fallback path in
``src/model/attention.py``.
"""
import itertools

import pytest
import torch

from src.model.config import GPTConfig
from src.model.transformer import GPT


def _make_config(pos_encoding: str, n_heads: int, n_kv_heads: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=16,
        d_model=32,
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=64,
        pos_encoding=pos_encoding,
        dropout=0.0,
        attn_dropout=0.0,
    )


@pytest.mark.parametrize(
    "pos_encoding,n_heads,n_kv_heads",
    list(itertools.product(["learned", "rope"], [4], [4, 2])),
)
def test_causal_mask_no_future_leakage(pos_encoding, n_heads, n_kv_heads):
    torch.manual_seed(0)
    cfg = _make_config(pos_encoding, n_heads, n_kv_heads)
    model = GPT(cfg)
    model.eval()

    B, T = 2, cfg.block_size
    tokens = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits_a, _, _ = model(tokens, targets=tokens)  # targets forces full-sequence logits

        tokens_b = tokens.clone()
        # Change only the very last token to a different value.
        tokens_b[:, -1] = (tokens_b[:, -1] + 1) % cfg.vocab_size
        logits_b, _, _ = model(tokens_b, targets=tokens_b)

    # Every position except the last must be unaffected by changing the
    # last token -- that's exactly what "causal" means.
    earlier_a = logits_a[:, :-1, :]
    earlier_b = logits_b[:, :-1, :]
    assert torch.allclose(earlier_a, earlier_b, atol=1e-5), (
        f"Future token leaked into earlier positions' attention "
        f"(pos_encoding={pos_encoding}, n_heads={n_heads}, n_kv_heads={n_kv_heads})"
    )

    # Sanity check the test itself isn't vacuous: the last position's
    # logits SHOULD generally differ once we change its own input token
    # embedding (not a hard guarantee for arbitrary random init, but true
    # with overwhelming probability).
    last_a = logits_a[:, -1, :]
    last_b = logits_b[:, -1, :]
    assert not torch.allclose(last_a, last_b, atol=1e-8)
