"""Abstract interface every tokenizer implementation must satisfy.

Keeping this tiny and explicit means the rest of the codebase (dataset
preparation, generation) never needs to know whether it's talking to the
byte-level BPE tokenizer in ``bpe.py`` or some other implementation dropped
in later (e.g. a WordPiece or SentencePiece tokenizer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Minimal encode/decode contract for a trainable text tokenizer."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct token ids this tokenizer can produce."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token ids."""

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token ids back into a string.

        Implementations must satisfy ``decode(encode(text)) == text`` for
        every string, including text containing characters unseen during
        training (this is why byte-level tokenization is used: it can
        always fall back to raw bytes for unknown content).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the tokenizer's learned vocabulary/merges to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Tokenizer":
        """Reconstruct a tokenizer previously written by ``save``."""
