"""Test encode/decode invertibility of the byte-level BPE tokenizer.

Because the tokenizer operates on raw UTF-8 bytes (not a fixed character
vocabulary), ``decode(encode(text)) == text`` must hold for *any* string,
including text containing characters the tokenizer never saw during
training -- that's the whole point of byte-level BPE over a word/character
vocabulary with an <unk> fallback.
"""
import pytest

from src.tokenizer.bpe import BPETokenizer

TRAIN_TEXT = (
    "the quick brown fox jumps over the lazy dog. " * 30
    + "she sells seashells by the seashore. " * 20
)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(TRAIN_TEXT, vocab_size=300, verbose=False)


@pytest.mark.parametrize("text", [
    "the quick brown fox",
    "",  # empty string
    "a",  # single character
    "12345 !@#$%^&*()",
    "Hello, World!\nNew line.\tTabbed.",
    "unseen words that were never in training data whatsoever",
    "Héllo wörld with áccents",         # Latin-1 supplement, unseen at train time
    "你好，世界！",                        # CJK, entirely unseen script
    "emoji test: 🚀🔥😀",                 # emoji / astral-plane codepoints
])
def test_roundtrip_invertibility(tokenizer, text):
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_vocab_size_matches_training_target(tokenizer):
    assert tokenizer.vocab_size == 300


def test_base_bytes_always_present(tokenizer):
    # The first 256 ids must always be the raw byte values, regardless of
    # what merges were learned.
    for i in range(256):
        assert tokenizer.vocab[i] == bytes([i])


def test_save_and_load_roundtrip(tokenizer, tmp_path):
    path = tmp_path / "vocab.json"
    tokenizer.save(str(path))
    reloaded = BPETokenizer.load(str(path))

    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.merges == tokenizer.merges

    text = "the quick brown fox jumps over the lazy dog."
    assert reloaded.decode(reloaded.encode(text)) == text
    assert reloaded.encode(text) == tokenizer.encode(text)


def test_encode_is_deterministic(tokenizer):
    text = "the quick brown fox jumps over the lazy dog"
    assert tokenizer.encode(text) == tokenizer.encode(text)
