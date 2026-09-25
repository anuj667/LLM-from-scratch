"""Minimal metrics logging: append-only JSON Lines file.

JSONL (one JSON object per line) is used instead of e.g. TensorBoard so
this repo has zero extra dependencies and the log can be parsed trivially
with `pandas.read_json(path, lines=True)` or a one-line list comprehension.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


class MetricsLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Start a fresh log file each run rather than silently appending to
        # a stale one from a previous, unrelated run.
        self._fh = open(path, "w", encoding="utf-8")

    def log(self, metrics: Dict[str, Any]) -> None:
        record = {"timestamp": time.time(), **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass
