# SpikeGPT-Myelin: Onboarding

A step-by-step guide to setting up the repos and running your first SpikeGPT
training run.

## What you are building

[`spikegpt-myelin`](https://github.com/djsaunde/spikegpt-myelin) is a
reproduction of the SpikeGPT spiking language model (enwik8 / WikiText). It
depends on [`myelin`](https://github.com/djsaunde/myelin), a spiking neural
network library that provides the core neuron kernels, surrogate gradients, and
Triton backends. You don't need to clone `myelin` yourself; the package manager
will pull it in automatically.

By the end of this guide you will have the environment installed and a small
training run finished.

## Prerequisites

| Requirement | Notes |
|---|---|
| Platform | **Linux x86_64 + NVIDIA GPU** for real training (Triton/CUDA kernels). **macOS (Apple Silicon)** works CPU-only for install, imports, and the tiny smoke run below (no Triton, no GPU training). On Windows, please use WSL2. |
| NVIDIA GPU (CUDA) | Required for real runs and all benchmarks. Not needed for the CPU smoke run. |
| Python 3.11 | Both repos pin 3.11. `uv` installs it for you if missing. |
| `git` | To clone the repo and (via `uv`) the `myelin` dependency. |
| `uv` | The package manager. Install with `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then restart your shell. |
| GitHub access | Both repos are **private**. You need read access to `djsaunde/spikegpt-myelin` **and** `djsaunde/myelin`. Ask Dan if you get a 404 when cloning. |

## Step 1: Authenticate to GitHub

`myelin` is pulled from a private git repo during install, so your machine needs
working GitHub credentials first, otherwise the install will fail on a silent
clone error.

The easiest path is the GitHub CLI:

```bash
# install gh (https://cli.github.com/), then:
gh auth login
gh auth setup-git   # makes git (and uv) use your gh credentials over HTTPS
```

Verify you can reach both private repos:

```bash
gh repo view djsaunde/spikegpt-myelin
gh repo view djsaunde/myelin
```

If either prints repo details, you're good.

## Step 2: Clone the training repo

```bash
git clone https://github.com/djsaunde/spikegpt-myelin.git
cd spikegpt-myelin
```

## Step 3: Install the environment

`uv sync` reads `pyproject.toml`, resolves everything (including `myelin` from
its git remote and the PyTorch 2.13 nightly build), and creates a `.venv`.

```bash
# CPU-only, enough to run the smoke test in Step 5
uv sync

# GPU box (recommended): add the CUDA/Triton kernels and W&B tracking
uv sync --extra cuda --extra tracking
```

Notes:
- The first sync downloads PyTorch nightly and clones `myelin`, so it can take a
  few minutes.
- torch 2.13 is required (the fused WKV recurrence needs `associative_scan`
  autograd, which only landed in 2.13). It is a nightly build, so it is pulled
  from the PyTorch nightly index.
- Optional extras you may want later: `--extra tokenization` (BPE / WikiText
  runs), `--extra app` (Streamlit playground), `--extra dev` (tests + linters).

## Step 4: Verify the install

```bash
uv run python -c "import spikegpt, myelin, torch; print('spikegpt + myelin import OK, torch', torch.__version__)"

# on a GPU box, confirm CUDA is visible:
uv run python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

If both imports succeed, the environment is ready.

## Step 5: Your first run (CPU smoke, ~1 minute)

This trains a tiny 111k-parameter model on a built-in toy string. It needs no
dataset and no GPU. Use it to confirm the whole training loop works end to end.

```bash
uv run python examples/train_tiny_spikegpt.py \
  --device cpu \
  --compile off \
  --context-length 32 \
  --layers 2 \
  --embedding 64 \
  --batch 16 \
  --steps 80 \
  --log-every 20 \
  --eval-every 20 \
  --sample-prompt spik \
  --sample-tokens 48
```

You should see the train loss drop from ~3.1 toward ~0.1 over 80 steps, periodic
validation BPC / perplexity, and a short sampled text snippet at the end. On
this toy corpus the model overfits fast (val loss rises again), which is
expected. The point is that the loop runs, not that the model is good.

## Step 6: Set up Weights & Biases (W&B)

We track every training run in [Weights & Biases](https://wandb.ai): loss, BPC,
spike rates, and sampled text. Please set it up before your first real run so
your experiments are logged and shareable.

1. Make a free account at [wandb.ai](https://wandb.ai).
2. Log in once (the `wandb` package comes from the `tracking` extra installed in
   Step 3):

```bash
uv run --extra tracking wandb login   # paste your key from https://wandb.ai/authorize
```

Then add `--wandb` to any training command to stream metrics to your dashboard.
`--wandb-project` groups runs (default `myelin`), `--wandb-run-name` labels one:

```bash
... --wandb --wandb-project spikegpt-<yourname> --wandb-run-name enwik8-first
```

## Step 7: A short GPU run on real data

This is the first "real" run: a small character-level model on enwik8, kept
short so it finishes quickly rather than chasing the paper result.

First, generate the dataset (not committed to the repo):

```bash
uv run python examples/prepare_token_corpus.py --help   # see available options
```

This writes the enwik8 data (into `data/` by convention). Then train a small
model for a few hundred steps:

```bash
uv run --extra tracking python examples/train_tiny_spikegpt.py \
  --device cuda \
  --text-file data/enwik8 --vocab byte \
  --context-length 256 --layers 4 --embedding 256 \
  --batch 16 --steps 500 \
  --lr 2e-3 --lr-final 1e-5 --warmup-steps 100 \
  --weight-decay 0.1 --amp bf16 --compile regional \
  --eval-every 100 \
  --wandb --wandb-project spikegpt-<yourname> --wandb-run-name enwik8-small \
  --best-checkpoint-out runs/enwik8_small.best.pt
```

Watch the validation BPC (bits per character) trend down. This short run will
not match the headline numbers, it is meant to give you a real loss curve and a
saved checkpoint to work with.

The run streams to W&B; open the run URL it prints at startup to watch the loss,
BPC, and spike-rate curves live. Drop the `--wandb*` flags to log to the console
only.

## Step 8 (optional): Evaluate a checkpoint

```bash
uv run python examples/evaluate_spikegpt_checkpoint.py runs/enwik8_small.best.pt \
  --text-file data/enwik8 --no-sample
```

This reports full-context strided BPC on the held-out tail.

## Reading the output

- **Train / Val Loss**: cross-entropy. Lower is better.
- **BPC** (bits per character): the headline metric for enwik8. The tuned full
  runs reach ~1.24; your short runs will be much higher, that is fine.
- **PPL** (perplexity): `exp(loss)`, another view of the same thing.
- **Spike Rate** (embedding / block): fraction of neurons that fired. This is a
  spiking network, so activations are sparse events rather than dense values.
  Roughly 0.1 to 0.5 is normal.
- **Step ms**: wall-clock per step. On GPU, expect the first few steps to be
  slow when `--compile regional` is on (Triton is compiling kernels), then it
  speeds up.

## Troubleshooting

- **`uv sync` fails cloning myelin / authentication error**: your GitHub
  credentials are not set up. Redo Step 1 (`gh auth setup-git`), then retry.
- **404 on `gh repo view`**: you do not have access to the private repo. Ask
  Dan.
- **On macOS**: `uv sync` installs a CPU-only torch (no Triton). Run only the
  CPU smoke (Step 5); `--device cuda`, benchmarks, and real training need a Linux
  x86_64 + NVIDIA GPU box. On Windows, use WSL2.
- **`torch.cuda.is_available()` is False on a GPU box**: check your NVIDIA
  driver. The nightly build targets CUDA 13; a very old driver will not work.
- **First GPU step hangs for a while**: that is `--compile regional` compiling
  Triton kernels. Wait it out, or drop to `--compile off` for quick iteration.

## Where to go next

- The full, paper-beating recipes (ctx-1024 and ctx-3072) are in the
  [spikegpt-myelin README](https://github.com/djsaunde/spikegpt-myelin). These
  are long runs (~9 to 15h on a single RTX 5090).
- `examples/` has evaluation, energy, and neuromorphic-feasibility scripts, plus
  a Streamlit playground under `examples/app/`.
- `docs/wkv_recurrence.md` explains the fused WKV time-mixing kernel.
- To understand the spiking primitives underneath, read the
  [myelin README](https://github.com/djsaunde/myelin) and its `examples/`
  (start with `train_mnist_rate.py`).
