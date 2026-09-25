"""Turn raw text into memory-mapped token arrays, and serve fixed-length
``(x, y)`` context windows for next-token prediction.

Two responsibilities live here:

1. ``prepare_bins`` / the ``--prepare`` CLI: tokenize a raw text file with a
   trained tokenizer and write it out as a flat, memory-mapped ``.bin`` file
   of ``uint16`` (or ``uint32`` for very large vocabularies) token ids --
   the same layout nanoGPT/GPT-2-style pipelines use, chosen because it lets
   training read random slices straight off disk via ``np.memmap`` without
   ever loading the whole (potentially huge) token array into RAM.

2. ``TokenDataset``: a ``torch.utils.data.Dataset`` that maps an index to a
   contiguous ``block_size``-token window ``x`` and the same window shifted
   by one token ``y`` (the next-token targets) -- the fundamental unit of
   supervision for autoregressive language modeling.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe import BPETokenizer


def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size <= 65535 else np.uint32


def prepare_bins(
    input_path: str,
    vocab_path: str,
    out_dir: str,
    val_fraction: float = 0.1,
) -> Tuple[str, str]:
    """Tokenize ``input_path`` with the tokenizer at ``vocab_path`` and
    write ``train.bin`` / ``val.bin`` memory-mapped token arrays into
    ``out_dir``. Returns their paths."""
    tok = BPETokenizer.load(vocab_path)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tok.encode(text)
    dtype = _dtype_for_vocab(tok.vocab_size)
    arr = np.array(ids, dtype=dtype)

    split = int(len(arr) * (1 - val_fraction))
    train_ids, val_ids = arr[:split], arr[split:]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Tokenized {len(text):,} chars -> {len(arr):,} tokens "
          f"(vocab_size={tok.vocab_size}, dtype={dtype.__name__})")
    print(f"  train: {len(train_ids):,} tokens -> {train_path}")
    print(f"  val:   {len(val_ids):,} tokens -> {val_path}")
    return train_path, val_path


class TokenDataset(Dataset):
    """Serves ``(x, y)`` context-window pairs from a memory-mapped token
    array on disk.

    Re-opening the memmap fresh in ``__getitem__`` (rather than once in
    ``__init__``) is deliberate: it avoids known issues with ``np.memmap``
    objects and PyTorch's multiprocessing ``DataLoader`` workers, where a
    memmap opened in the parent process can misbehave once forked.
    """

    def __init__(self, bin_path: str, block_size: int, vocab_size: int):
        self.bin_path = bin_path
        self.block_size = block_size
        self.dtype = _dtype_for_vocab(vocab_size)
        # Just to know the length up front; the actual reads happen lazily.
        self._n_tokens = os.path.getsize(bin_path) // np.dtype(self.dtype).itemsize

    def __len__(self) -> int:
        # Every starting offset from 0 up to n_tokens - block_size - 1
        # yields a valid (x, y) window.
        return max(0, self._n_tokens - self.block_size - 1)

    def __getitem__(self, idx: int):
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        x = torch.from_numpy(data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(data[idx + 1: idx + 1 + self.block_size].astype(np.int64))
        return x, y


def _main():
    parser = argparse.ArgumentParser(description="Tokenize a corpus into train/val .bin files")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.prepare:
        prepare_bins(args.input, args.vocab, args.out_dir, args.val_fraction)


if __name__ == "__main__":
    _main()
