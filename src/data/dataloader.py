"""Batching utilities for the memory-mapped token dataset.

Provides a thin ``build_dataloader`` wrapper around ``torch.utils.data.DataLoader``
plus a dependency-free ``RandomBatchSampler`` used by the trainer for quick,
allocation-light random-window sampling (the standard nanoGPT-style
``get_batch`` pattern), which is faster than going through the full Dataset
+ DataLoader machinery when you just want random fixed-length windows
straight off a memmap.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TokenDataset, _dtype_for_vocab


def build_dataloader(
    bin_path: str,
    block_size: int,
    vocab_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokenDataset(bin_path, block_size=block_size, vocab_size=vocab_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def get_random_batch(
    bin_path: str,
    block_size: int,
    vocab_size: int,
    batch_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Directly sample a random batch of (x, y) windows from a memmap,
    without going through the Dataset/DataLoader abstraction. This is the
    "get_batch" pattern popularized by nanoGPT: for a single flat token
    array it's simpler and faster than a full DataLoader, since there's no
    meaningful notion of "epoch" over a single long token stream anyway.
    """
    dtype = _dtype_for_vocab(vocab_size)
    data = np.memmap(bin_path, dtype=dtype, mode="r")
    max_start = len(data) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = torch.stack([
        torch.from_numpy(data[s: s + block_size].astype(np.int64)) for s in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[s + 1: s + 1 + block_size].astype(np.int64)) for s in starts
    ])

    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y
