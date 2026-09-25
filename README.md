# llm-from-scratch

A GPT-style, decoder-only Transformer language model — **tokenizer, architecture, training loop, and text generation, all built from scratch** in plain PyTorch and NumPy. No Hugging Face, no pre-built transformer libraries. Every matrix multiplication in the attention mechanism, every line of the BPE tokenizer, and every step of the training loop is written out and readable.

```
raw text  →  byte-level BPE tokenizer  →  binary token shards  →  Transformer  →  pretraining loop  →  sampling / generation
```

This project exists to answer one question properly: **what actually happens between typing a prompt and a language model generating text?** Every design decision below is deliberate and explained, not just copied from a tutorial.

---

## Features

| Component | What's implemented |
|---|---|
| **Tokenizer** | Byte-level Byte-Pair Encoding (BPE), trained from scratch — no `<unk>` fallback needed, since every byte is representable |
| **Attention** | Causal multi-head self-attention from raw matmuls, with optional **Grouped-Query** / **Multi-Query Attention** |
| **Positional encoding** | Learned absolute embeddings *or* **RoPE** (Rotary Position Embeddings) — configurable per run |
| **Normalization** | LayerNorm *or* **RMSNorm** |
| **Feedforward** | Standard GeLU MLP *or* **SwiGLU** (LLaMA-style gated activation) |
| **Training** | AdamW with correct weight-decay exclusions, cosine LR schedule with linear warmup, gradient clipping, mixed precision (AMP) |
| **Inference** | KV-cached autoregressive generation with temperature / top-k / top-p sampling |
| **Checkpointing** | Self-describing checkpoints (architecture + weights + optimizer state bundled together) with resume support |
| **Tests** | Causal-mask leakage tests, tensor-shape tests, tokenizer round-trip invertibility tests |

---

## Project structure

```
llm-from-scratch/
├── data/
│   ├── raw/corpus.txt            # your training text goes here
│   └── processed/                # tokenized train.bin / val.bin (generated)
├── configs/
│   ├── gpt_nano.yaml             # ~3-10M params — fast CPU iteration
│   ├── gpt_small.yaml            # 124M params, GPT-2-small-shaped — needs a GPU
│   └── training_config.yaml      # annotated reference for every training hyperparameter
├── src/
│   ├── tokenizer/                # byte-level BPE: base.py, bpe.py
│   ├── model/                    # embedding, attention, feedforward, norm, block, transformer
│   ├── data/                     # dataset preparation + memory-mapped batching
│   ├── training/                 # trainer, optimizer, LR scheduler
│   ├── inference/                # generator, sampler, KV cache
│   └── utils/                    # logging, checkpointing
├── tests/                        # causal-mask, shape, and tokenizer tests
├── notebooks/                    # interactive walkthroughs of each subsystem
├── pretrain.py                   # entry point: train a model
├── generate.py                   # entry point: generate text from a checkpoint
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/llm-from-scratch.git
cd llm-from-scratch

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Train a tokenizer on your text**
```bash
python -m src.tokenizer.bpe --train --input data/raw/corpus.txt \
    --vocab-size 2000 --out src/tokenizer/vocab.json
```

**2. Tokenize the corpus into training data**
```bash
python -m src.data.dataset --prepare --input data/raw/corpus.txt \
    --vocab src/tokenizer/vocab.json --out-dir data/processed
```

**3. Make sure `vocab_size` in your config matches step 1's `--vocab-size` exactly**, then pretrain:
```bash
python pretrain.py --config configs/gpt_nano.yaml
```

**4. Generate text from a checkpoint**
```bash
python generate.py --checkpoint checkpoints/step_2000.pt \
    --vocab src/tokenizer/vocab.json \
    --prompt "Once upon a time" \
    --temperature 0.8 --top-k 40
```

**Resuming an interrupted or extended run** (raise `max_steps` in your config first):
```bash
python pretrain.py --config configs/gpt_nano.yaml --resume checkpoints/step_2000.pt
```

---

## Design notes

**Tokenizer — byte-level BPE.** Operating on raw UTF-8 bytes (256 base tokens) rather than a fixed character/word vocabulary means the tokenizer can represent *any* input, including text and scripts it never saw during training, with no information loss. `decode(encode(text)) == text` always holds.

**Positional encoding — learned vs. RoPE.** Learned embeddings (GPT-2 style) are simple but don't generalize past the trained context length. RoPE rotates query/key vectors by an angle proportional to position, baking relative-position information directly into the attention dot product — used by most modern LLMs (LLaMA, Mistral, etc.) for better length extrapolation.

**Attention — MHA, GQA, and MQA.** `n_kv_heads == n_heads` gives standard multi-head attention. Setting `n_kv_heads < n_heads` shares key/value heads across multiple query heads (Grouped-Query Attention), shrinking the KV cache with minimal quality loss — the same trick used in production models to make long-context inference affordable.

**Feedforward — GeLU vs. SwiGLU.** A plain two-layer GeLU MLP is the GPT-2 default. SwiGLU (LLaMA/PaLM/Mistral) gates one projection with SiLU before multiplying it into another, which empirically outperforms GeLU at matched parameter count.

**Pre-norm residual stream.** Every block normalizes *before* the sublayer (`norm → attention/FFN → add residual`) rather than after. This keeps the residual stream numerically well-behaved through depth, which is what makes deep Transformers trainable without exotic initialization tricks.

**Training recipe.** Linear warmup avoids unstable early updates while Adam's moment estimates are poorly calibrated; cosine decay afterward gives the best final loss for a fixed compute budget — the same schedule shape used by GPT-2/GPT-3 and nearly every LLM pretraining run since.

---

## Tests

```bash
pytest tests/ -v
```

- **`test_causal_mask.py`** — verifies that changing a future token can *never* change an earlier position's logits. This is the single most important property of an autoregressive model, and the easiest thing to silently break.
- **`test_shapes.py`** — walks a batch through every module and pins down the exact expected tensor shape at each stage, including a check that cached, token-by-token decoding produces *identical* logits to a full forward pass.
- **`test_tokenizer.py`** — checks encode/decode invertibility on ASCII, accented Latin, CJK, and emoji text, including strings never seen during training.

---

## Model sizes

| Config | Params | Use case |
|---|---|---|
| `gpt_nano.yaml` | ~3–10M | Fast local iteration, CPU-friendly, sanity checks |
| `gpt_small.yaml` | ~124M | GPT-2-small-shaped; needs a GPU and a much larger corpus/vocab |

Both share identical code — only the YAML config changes. Swapping `pos_encoding`, `norm_type`, or `activation` in either is a one-line change.

---

## What this is *not*

An educational, single-machine implementation. It intentionally does not include distributed/multi-GPU training, fused/FlashAttention kernels, or dataset deduplication pipelines — all essential at real scale, but out of scope for what this repo is trying to teach.

---

## License

[Choose a license — MIT is a common permissive default for educational ML projects.]
