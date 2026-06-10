# spikegpt-myelin

A SpikeGPT reproduction on enwik8 / WikiText, built on the
[myelin](https://github.com/djsaunde/myelin) spiking-neural-network library.

Core spiking-neuron kernels, surrogate gradients, packing, and hardware export
live in `myelin` (a dependency). This package adds the SpikeGPT/RWKV language
model, the fused WKV time-mixing recurrence, the memory-mapped token corpus, and
spike statistics — everything specific to training and evaluating the language
model.

## Result

We reproduce — and beat — the SpikeGPT paper's enwik8 result with a 41M model
(12 layers / 512 embd) on a single consumer RTX 5090. All numbers are
full-context strided eval (BPC) on the held-out last 5M bytes.

| Setting | Our test BPC | Paper | Recipe / cost |
|---|---:|---:|---|
| **ctx-1024 (tuned)** | **1.235** | 1.283 | batch 64, lr `2e-3`, ~9h — beats the paper |
| ctx-1024 (batch-12 repro) | 1.281 | 1.283 | full ~10B-token budget, ~15h — a match |
| **ctx-3072 (headline)** | **1.239** | 1.262 | batch 24, lr `8e-4`, ctx 3072 — beats the paper |

The tuned ctx-1024 run is both faster and better than the faithful batch-12
reproduction (a larger batch with a correspondingly larger LR finds a better
optimum in ~60% of the wall-clock). We use a stabilized recipe (cosine LR decay,
weight decay `0.1`, bf16) rather than the paper's literal `6e-4` / `wd 0`, which
diverges in our setup. The split is the standard disjoint enwik8 split — train
(first 90M), val (90–95M, drives selection), test (last 5M) — with deterministic
full-context strided eval, so a run yields a genuinely held-out,
selection-noise-free number. The WKV recurrence runs through a fused Triton
custom op on CUDA (sequential-over-time; ~5–9x faster than chunked/parallel
matmul forms — see `benchmarks/results/wkv_throughput_rtxpro6000.md` and
`docs/wkv_recurrence.md`).

## Setup

```bash
uv sync --extra cuda --extra tracking
```

This pulls `myelin` from its git remote and torch/torchvision/triton from the
PyTorch nightly CUDA index (torch 2.13 is required — the WKV path uses
`associative_scan`, which only has correct autograd in 2.13+). Add
`--extra tokenization` for BPE/WikiText runs, `--extra app` for the Streamlit
playground, and `--extra dev` for the test/lint toolchain.

Datasets (`data/enwik8`, WikiText) are not committed; regenerate them with
`examples/prepare_token_corpus.py`.

## Train and evaluate

```bash
# train the tuned ctx-1024 model (best test BPC, ~9h on one 5090).
# Clean disjoint split: train 0-90M, val 90-95M (drives selection), test 95-100M.
uv run --extra tracking python examples/train_tiny_spikegpt.py \
  --text-file data/enwik8 --vocab byte --context-length 1024 --layers 12 --embedding 512 \
  --test-tokens 5000000 --min-val-tokens 5000000 --val-fraction 0.0 \
  --batch 64 --steps 156000 --lr 2e-3 --lr-final 1e-5 --warmup-steps 2000 \
  --weight-decay 0.1 --dropout 0.03 --amp bf16 --compile regional \
  --best-checkpoint-out runs/enwik8_fast.best.pt

# evaluate on the untouched test tail (full-context BPC by default)
uv run python examples/evaluate_spikegpt_checkpoint.py runs/enwik8_fast.best.pt \
  --text-file data/enwik8_test --no-sample
```

## Layout

- `src/spikegpt/language.py` — SpikeGPT/RWKV model, vocabularies, sampling, eval.
- `src/spikegpt/wkv_triton.py`, `wkv_bf16.py` — fused WKV recurrence custom ops.
- `src/spikegpt/token_corpus.py` — memory-mapped token corpus for pretraining.
- `src/spikegpt/spike_statistics.py` — dead/saturated-neuron and density analysis.
- `src/spikegpt/benchmarks/` — training/generation/MFU/WKV-throughput benchmarks
  (run with `python -m spikegpt.benchmarks.<name>`).
- `examples/` — training, evaluation, energy, neuromorphic-feasibility scripts and
  the Streamlit playground (`examples/app/`).
- `modal/` — cloud training launchers.
- `docs/` — WKV recurrence rationale, neuromorphic deployment notes.

## Other reproductions

- **216M WikiText-103**: test PPL 26.47 vs paper 39.75. See
  `benchmarks/results/spikegpt_216m_wikitext.md`.
- **Neuromorphic feasibility** (SpiNNaker 2): see
  `docs/neuromorphic_deployment.md` and
  `examples/spikegpt_neuromorphic_feasibility.py`.
