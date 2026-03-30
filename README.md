# CUSTOM_JLX

A custom sparse ternary LLM training framework built from scratch on Apple Silicon, using Metal GPU kernels.

## What this is

An experiment in training large language models under extreme memory constraints.

- Ternary weights (+1 / 0 / −1) with 1% sparsity (CSR/CSC sparse format)
- Custom Metal GPU kernels for sparse forward/backward passes
- HiMA (Hierarchical Memory-Aware training) — dynamically tiers weight blocks into Hot / Warm / Cold based on gradient magnitude
- Custom Adafactor optimizer with 1-bit momentum compression

. 1B–7B scale is practical today.

## What this is not

- Production-ready code
- Clean code (`main.mm` is 972 lines and I'm not sorry)
- Cross-platform (Metal only, Apple Silicon only)

## Build

```bash
python scripts/prepare_data.py

make
./custom_jlx
```

Edit `config.txt` to adjust model size, sparsity, and training parameters.

## Why I'm sharing this

Someone told me this was garbage theory and garbage code.

The overfit test disagreed. Make of that what you will.

If you find this interesting — the HiMA memory tiering idea, the sparse ternary approach, anything — I'd genuinely like to hear from you. Looking for people to think about this with.

## Status

Proof of concept. No guarantees. PRs welcome.