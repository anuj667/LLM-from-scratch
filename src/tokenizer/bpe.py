"""Byte-level Byte-Pair Encoding (BPE), implemented from scratch.

This is the same core idea GPT-2 uses: operate on raw UTF-8 bytes (256
possible base tokens) rather than Unicode characters, so the tokenizer can
represent *any* input -- there is no "unknown token" fallback needed, and
``decode(encode(text)) == text`` holds for every string.

Algorithm
---------
1. Pre-tokenize text into chunks (words/punctuation/whitespace runs) with a
   regex, so merges never cross a word boundary (this keeps "dog" and
   "dog." from being treated totally differently, and keeps the vocabulary
   from wasting merges on cross-word garbage).
2. Represent each chunk as a tuple of byte values (0-255).
3. Repeatedly find the most frequent adjacent pair of tokens across the
   whole corpus and merge it into a new token id. Record merges in the
   order they were learned -- that order is also the priority used at
   encode time.
4. Stop once the vocabulary reaches ``vocab_size`` or no pair occurs more
   than once.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

from src.tokenizer.base import Tokenizer

# A simplified version of GPT-2's pre-tokenization regex: splits text into
# words (with a leading space kept attached, GPT-2 style), numbers,
# punctuation runs, and whitespace runs. Python's stdlib `re` module has no
# `\p{L}` Unicode-property support (that requires the third-party `regex`
# package), so this uses `\w`/`\W` under `re.UNICODE`, which Python already
# treats as "any Unicode letter/digit" vs. "everything else" -- good enough
# for pre-tokenization purposes and dependency-free.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pre_tokenize(text: str) -> List[str]:
    return _SPLIT_PATTERN.findall(text)


def _get_pair_counts(sequences: List[List[int]]) -> Counter:
    counts: Counter = Counter()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    return counts


def _merge_pair(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    merged = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged


class BPETokenizer(Tokenizer):
    """Byte-level BPE tokenizer trained from scratch on a text corpus."""

    def __init__(self, merges: List[Tuple[int, int]] | None = None,
                 vocab: Dict[int, bytes] | None = None):
        # `merges` is the ordered list of (id_a, id_b) pairs that were
        # merged during training. Order matters: at encode time we always
        # apply the *earliest-learned* applicable merge first, exactly
        # mirroring the greedy order used during training.
        self.merges: List[Tuple[int, int]] = merges or []
        self.merge_ranks: Dict[Tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        # `vocab` maps token id -> the raw bytes it represents. The first
        # 256 ids are always the raw byte values.
        self.vocab: Dict[int, bytes] = vocab or {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the base byte tokens)")

        chunks = _pre_tokenize(text)
        # Represent every chunk as a list of byte-value ints.
        sequences: List[List[int]] = [list(chunk.encode("utf-8")) for chunk in chunks]

        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: List[Tuple[int, int]] = []
        next_id = 256
        num_merges = vocab_size - 256

        for step in range(num_merges):
            pair_counts = _get_pair_counts(sequences)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # no repeated pair left worth merging

            sequences = [_merge_pair(seq, best_pair, next_id) for seq in sequences]
            vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges.append(best_pair)
            if verbose and (step % 50 == 0 or step == num_merges - 1):
                print(f"merge {step + 1}/{num_merges}: {best_pair} -> {next_id} "
                      f"(count={best_count}, token={vocab[next_id]!r})")
            next_id += 1

        tok = cls(merges=merges, vocab=vocab)
        return tok

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # Find the pair in this sequence with the lowest merge rank
            # (i.e. the one learned earliest during training).
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])
            new_id = 256 + self.merge_ranks[best_pair]
            ids = _merge_pair(ids, best_pair, new_id)
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for chunk in _pre_tokenize(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def decode(self, ids: List[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            # merges stored as [a, b] pairs, in learned order
            "merges": [list(p) for p in self.merges],
            # vocab stored as id -> hex string of raw bytes (JSON can't
            # hold arbitrary binary data directly)
            "vocab": {str(i): b.hex() for i, b in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(p) for p in data["merges"]]
        vocab = {int(i): bytes.fromhex(h) for i, h in data["vocab"].items()}
        return cls(merges=merges, vocab=vocab)


def _main():
    parser = argparse.ArgumentParser(description="Train / test the byte-level BPE tokenizer")
    parser.add_argument("--train", action="store_true", help="train a new tokenizer")
    parser.add_argument("--input", type=str, required=True, help="path to raw text corpus")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--out", type=str, default="src/tokenizer/vocab.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if args.train:
        tok = BPETokenizer.train(text, vocab_size=args.vocab_size)
        tok.save(args.out)
        print(f"Trained tokenizer with vocab_size={tok.vocab_size}, "
              f"saved to {args.out}")

        # Round-trip sanity check on a sample.
        sample = text[:200]
        ids = tok.encode(sample)
        back = tok.decode(ids)
        assert back == sample, "round-trip encode/decode failed!"
        print(f"Round-trip check passed on {len(sample)} chars "
              f"({len(ids)} tokens).")


if __name__ == "__main__":
    _main()
