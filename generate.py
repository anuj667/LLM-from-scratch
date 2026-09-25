#!/usr/bin/env python3
"""Generate text from a trained checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/step_5000.pt \
        --vocab src/tokenizer/vocab.json \
        --prompt "Once upon a time" \
        --max-new-tokens 200 --temperature 0.8 --top-k 40
"""
from __future__ import annotations

import argparse

import torch

from src.inference.generator import generate
from src.tokenizer.bpe import BPETokenizer
from src.utils.checkpoint import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, ckpt = load_model_from_checkpoint(args.checkpoint, map_location=args.device)
    tokenizer = BPETokenizer.load(args.vocab)
    print(f"Loaded model from step {ckpt.get('step', '?')} "
          f"({model.num_params():,} non-embedding params)")

    text = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        device=args.device,
    )
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
