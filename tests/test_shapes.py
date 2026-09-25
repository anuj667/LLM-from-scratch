"""Verify tensor dimensions at every stage of the forward pass.

Shape bugs are the most common way a from-scratch Transformer silently
produces garbage (e.g. a transpose that broadcasts instead of erroring, or
a reshape that quietly scrambles the head dimension). These tests pin down
the exact expected shape after each module.
"""
import pytest
import torch

from src.model.attention import CausalSelfAttention
from src.model.block import TransformerBlock
from src.model.config import GPTConfig
from src.model.embedding import TokenEmbedding, apply_rope, precompute_rope_cache
from src.model.feedforward import build_feedforward
from src.model.transformer import GPT

B, T = 3, 10


def _cfg(**overrides) -> GPTConfig:
    base = dict(
        vocab_size=100, block_size=16, d_model=32, n_layers=2,
        n_heads=4, n_kv_heads=4, d_ff=64, dropout=0.0, attn_dropout=0.0,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_token_embedding_shape():
    cfg = _cfg()
    embed = TokenEmbedding(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    out = embed(tokens)
    assert out.shape == (B, T, cfg.d_model)


def test_rope_cache_and_apply_shapes():
    head_dim = 8
    cos, sin = precompute_rope_cache(head_dim, max_seq_len=20)
    assert cos.shape == (20, head_dim // 2)
    assert sin.shape == (20, head_dim // 2)

    x = torch.randn(B, 4, T, head_dim)  # (B, n_heads, T, head_dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


@pytest.mark.parametrize("n_heads,n_kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_attention_shape_mha_and_gqa(n_heads, n_kv_heads):
    cfg = _cfg(n_heads=n_heads, n_kv_heads=n_kv_heads)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out, present_kv = attn(x, use_cache=True)
    assert out.shape == (B, T, cfg.d_model)
    k, v = present_kv
    assert k.shape == (B, n_kv_heads, T, cfg.head_dim)
    assert v.shape == (B, n_kv_heads, T, cfg.head_dim)


def test_feedforward_shapes_gelu_and_swiglu():
    for activation in ("gelu", "swiglu"):
        cfg = _cfg(activation=activation)
        ffn = build_feedforward(cfg)
        x = torch.randn(B, T, cfg.d_model)
        out = ffn(x)
        assert out.shape == (B, T, cfg.d_model)


def test_transformer_block_shape():
    cfg = _cfg()
    block = TransformerBlock(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out, present_kv = block(x, use_cache=True)
    assert out.shape == (B, T, cfg.d_model)


@pytest.mark.parametrize("pos_encoding", ["learned", "rope"])
def test_full_model_forward_with_and_without_targets(pos_encoding):
    cfg = _cfg(pos_encoding=pos_encoding)
    model = GPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss, _ = model(tokens, targets=tokens)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss.dim() == 0  # scalar

    logits_infer, loss_none, _ = model(tokens)
    assert logits_infer.shape == (B, 1, cfg.vocab_size)  # only last position
    assert loss_none is None


def test_full_model_kv_cache_matches_full_forward():
    """The output of decoding token-by-token with a KV cache must match
    running the whole sequence through in one shot (up to floating point
    tolerance) -- this is the property that makes the KV cache a pure
    speed optimization rather than a behavior change."""
    torch.manual_seed(0)
    cfg = _cfg(pos_encoding="rope")
    model = GPT(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full_logits, _, _ = model(tokens, targets=tokens)

        # Now replay the same sequence one token at a time through the cache.
        past_kv = None
        cached_logits = []
        for t in range(T):
            tok = tokens[:, t: t + 1]
            logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True)
            cached_logits.append(logits[:, -1, :])
        cached_logits = torch.stack(cached_logits, dim=1)  # (1, T, vocab)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4)
