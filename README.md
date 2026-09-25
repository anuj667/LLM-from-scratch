# llm-from-scratch

A minimal, readable, end-to-end implementation of a decoder-only Transformer
language model (GPT-style), trained the way real pre-training pipelines work:

```
raw text -> BPE tokenizer -> binary token shards -> Transformer -> pretraining loop -> sampling/generation
```

Nothing here is a wrapper around someone else's library. The tokenizer, the
attention mechanism, the training loop, and the sampler are all implemented
from scratch in plain PyTorch + NumPy, so you can read every line and know
exactly what it does. It's small enough to train a real (if tiny) model on a
laptop CPU in a few minutes, and structured so that swapping in a bigger
config (`configs/gpt_small.yaml`) and a real GPU gets you a genuine
124M-parameter GPT-2-shaped model.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Put raw text in data/raw/corpus.txt (a sample is already there)

# 2. Train a byte-level BPE tokenizer on it and tokenize the corpus into
#    train.bin / val.bin
python -m src.tokenizer.bpe --train --input data/raw/corpus.txt \
    --vocab-size 512 --out src/tokenizer/vocab.json
python -m src.data.dataset --prepare --input data/raw/corpus.txt \
    --vocab src/tokenizer/vocab.json --out-dir data/processed

# 3. Pretrain a tiny model
python pretrain.py --config configs/gpt_nano.yaml

# 4. Generate text from a checkpoint
python generate.py --checkpoint checkpoints/step_5000.pt \
    --vocab src/tokenizer/vocab.json --prompt "Once upon a time"
```

## Layout

See the project tree below for what lives where; every subpackage under
`src/` has a single, focused job:

- `src/tokenizer` — byte-level BPE, trained from scratch on your corpus.
- `src/model` — the Transformer itself: embeddings, causal attention
  (supports Grouped-Query Attention), SwiGLU/GeLU feedforward, RMSNorm or
  LayerNorm, and the decoder block that wires them together with residuals.
- `src/data` — turns raw text into memory-mapped token arrays and serves
  fixed-length `(x, y)` context windows for next-token prediction.
- `src/training` — the training loop itself: AdamW with weight-decay
  exclusions, cosine LR schedule with linear warmup, gradient clipping, and
  mixed precision.
- `src/inference` — autoregressive generation with temperature / top-k /
  top-p sampling and a KV cache for fast decoding.
- `src/utils` — logging and checkpointing.
- `tests/` — shape and causality tests that catch the two bugs that ruin
  every from-scratch Transformer implementation: leaking future tokens
  through the attention mask, and silently broken tensor shapes.

## Design notes

- **Tokenizer**: byte-level BPE (same idea as GPT-2's), so it never hits an
  unknown token — every byte is representable — and the vocabulary is
  learned entirely from your data rather than imported from elsewhere.
- **Positional info**: `src/model/embedding.py` supports both learned
  absolute positional embeddings (simplest, GPT-2 style) and RoPE (rotary
  embeddings, used by most modern LLMs); pick one in the model config.
- **Attention**: `src/model/attention.py` implements causal multi-head
  attention from the underlying matrix multiplications (not
  `nn.MultiheadAttention`), and generalizes to Grouped-Query Attention by
  letting the number of key/value heads be smaller than the number of query
  heads.
- **Configs**: `configs/gpt_nano.yaml` (~10M params) is meant for fast
  iteration and CPU sanity checks; `configs/gpt_small.yaml` is a
  GPT-2-small-shaped 124M param config for when you have a GPU.

## Tests

```bash
pytest tests/
```

`test_causal_mask.py` verifies that no position can attend to a future
position (by checking that changing a future token never changes the logits
at earlier positions). `test_shapes.py` walks a batch through every module
and checks tensor shapes at each stage. `test_tokenizer.py` checks that
`decode(encode(text)) == text` for arbitrary Unicode input, including
characters outside the training data (byte-level BPE guarantees this).

## What this is not

This is an educational, single-machine implementation. It does not include
distributed/multi-GPU training, FlashAttention kernels, or dataset
deduplication/filtering pipelines — all of which matter enormously at real
scale but would obscure the core ideas this repo is meant to teach.
