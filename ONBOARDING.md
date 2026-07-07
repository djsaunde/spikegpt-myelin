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
| Platform | **Linux x86_64 + NVIDIA GPU** for real training (Triton/CUDA kernels). **macOS (Apple Silicon)** works CPU-only for install, imports, and the tiny smoke run below (no Triton, no GPU training). On Windows, please use WSL2. **No local GPU?** See [Running on rented GPUs (Lambda Cloud)](#running-on-rented-gpus-lambda-cloud) to run the GPU steps from your laptop. |
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

This is the first "real" run: a small byte-level model on enwik8, kept short so
it finishes quickly rather than chasing the paper result.

First, get the dataset (not committed to the repo). enwik8 is the first 100 MB of
English Wikipedia; download it once and carve off the held-out test tail (the
last 5 M bytes, matching the paper's train/val/test split):

```bash
mkdir -p data
curl -fsSL https://mattmahoney.net/dc/enwik8.zip -o data/enwik8.zip
cd data && unzip -q enwik8.zip && rm enwik8.zip && cd ..   # -> data/enwik8 (100,000,000 bytes)
tail -c 5000000 data/enwik8 > data/enwik8_test             # last 5 M bytes = test split
```

Then train a small model for a few hundred steps. The training script makes the
disjoint train (first 90 M) / val (next 5 M) / test (last 5 M) split internally
from `data/enwik8`, so the test tail is never trained on:

```bash
uv run --extra tracking python examples/train_tiny_spikegpt.py \
  --device cuda \
  --text-file data/enwik8 --vocab byte \
  --context-length 256 --layers 4 --embedding 256 \
  --test-tokens 5000000 --min-val-tokens 5000000 --val-fraction 0.0 \
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
  --text-file data/enwik8_test --no-sample
```

This reports full-context strided BPC on the held-out test tail (`data/enwik8_test`,
the 5 M bytes never seen during training).

## Bigger datasets (optional)

enwik8 is byte-level. For subword (BPE) pretraining on more data, tokenize a
streaming HuggingFace dataset into a `.bin` corpus first. `--dataset` gives you
named choices — `fineweb-edu` (educational-quality web text, defaults to the
10B-token sample) and `openwebtext`:

```bash
uv run --extra tokenization python examples/prepare_token_corpus.py \
  --dataset fineweb-edu --max-tokens 1_000_000_000 --output data/fineweb_edu_1b.bin
```

Then train from it with `--train-bin data/fineweb_edu_1b.bin --vocab bpe` instead
of `--text-file ... --vocab byte`. (Pass `--hf-config sample-100BT` for a larger
FineWeb-Edu slice.)

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

## Running on rented GPUs (Lambda Cloud)

No local NVIDIA GPU (e.g. you're on a Mac)? You can run the GPU steps above —
Step 7 training, benchmarks, evaluation — on a rented [Lambda
Cloud](https://lambda.ai) H100 straight from your laptop. The `lambda/` harness
launches an instance, replicates *this exact environment* on it, runs your
command, copies results back, and — importantly — **terminates the instance
afterward so you stop paying**.

### One-time setup

1. **Lambda account + credits.** Sign up at [lambda.ai](https://lambda.ai), add
   credits, and confirm your account has H100 quota (new accounts may need to
   request GPU access). Create an API key at
   <https://cloud.lambdalabs.com/api-keys>.
2. **Give the harness your key.** Export it in your shell (or, if you use
   [direnv](https://direnv.net), put it in a gitignored `.envrc`):
   ```bash
   export LAMBDA_API_KEY=secret_...        # from the api-keys page
   ```
3. **GitHub + SSH.** The harness reuses the `gh` auth from Step 1 to clone the
   private `myelin` repo onto the VM — no extra token setup. Register your SSH
   public key with Lambda once (defaults to `~/.ssh/id_ed25519`; pass
   `--key <path>` if yours is elsewhere):
   ```bash
   python3 lambda/run.py keys --add
   ```

The driver is stdlib-only — run it with your system `python3`, no venv needed.

### Smoke test (confirm it all works, ~5–8 min, ~$0.60)

```bash
python3 lambda/run.py smoke
```

This launches an H100, replicates the env (installs `uv`, syncs the frozen CUDA
env, clones `myelin`, and adds NVIDIA CUDA-13 forward-compat so the cu130 torch
runs on Lambda's driver), runs a torch + Triton-WKV kernel check on the GPU, then
terminates. Success looks like the torch/triton versions, `device NVIDIA H100 …`,
and `triton WKV ok`.

### Run an experiment

`run -- <cmd>` and `train` launch, provision, run, copy `runs/` back, then
terminate. The `data/enwik8` you built in Step 7 is rsynced up automatically, so
training on real data just works:

```bash
# the Step-7 short run, but on a rented H100:
python3 lambda/run.py train --layers 4 --embedding 256 --context-length 256 \
  --batch 16 --steps 500 \
  -- --test-tokens 5000000 --min-val-tokens 5000000 --val-fraction 0.0 \
     --best-checkpoint-out runs/enwik8_small.best.pt

# or any command in the synced env:
python3 lambda/run.py run -- python -m spikegpt.benchmarks.wkv_throughput --device cuda
```

**Flag placement matters.** Everything *after* `--` is the command that runs on the
VM; harness flags (`--type`, `--extra`, `--region`, `--name`, `--keep`, …) must come
*before* it. A harness flag placed after `--` is silently swallowed into the command
and ignored — e.g. a stray `--type` after `--` falls back to the default GPU (see
below), so keep them before the separator:

```bash
python3 lambda/run.py run \
  --type gpu_1x_a100_sxm4 --extra "cuda --extra tokenization" \
  -- \
  python examples/train_tiny_spikegpt.py --device cuda --vocab bpe \
  --train-bin data/fineweb_edu_600m.bin --layers 4 --embedding 128 ...
```

**Choosing the GPU.** `--type` selects the instance type; it defaults to
`gpu_1x_h100_sxm5`. Lambda capacity comes and goes, so check what's actually
available before launching and pass a type that has capacity:

```bash
python3 lambda/run.py types            # all types + which regions have capacity
python3 lambda/run.py types --gpu a100 # filter, e.g. gpu_1x_a100_sxm4
```

### Iterating: keep one instance up

The heavy env sync runs on every fresh VM (several minutes). For a work session,
provision one instance once and reuse it:

```bash
python3 lambda/run.py launch --type gpu_1x_a100_sxm4 --name spikegpt-dev  # prints  <id>  <ip>
python3 lambda/run.py provision <id>                 # env sync, once
python3 lambda/run.py exec <id> -- python examples/train_tiny_spikegpt.py --device cuda ...
python3 lambda/run.py fetch <id> runs/ runs/         # pull checkpoints/artifacts down
python3 lambda/run.py terminate <id>                 # WHEN DONE — stops billing
```

`exec` flags (`--extra`, `--key`) must come *before* the `<id>` (it's a
positional) — anything after the id is swallowed into the remote command, so
`--extra` there is silently ignored and falls back to `cuda`:

```bash
python3 lambda/run.py exec --extra tokenization <id> -- python ...   # ✅ extra applied
python3 lambda/run.py exec <id> --extra tokenization -- python ...   # ❌ ignored (uses cuda)
```

For the BPE path ([Bigger datasets](#bigger-datasets-optional)), tokenize *on
the instance* rather than uploading a multi-GB `.bin` — `exec` doesn't rsync, so
a corpus built by one `exec` persists for the next (don't train via one-shot
`run` afterward; its `--delete` rsync would wipe it):

```bash
python3 lambda/run.py exec --extra tokenization <id> -- \
  python examples/prepare_token_corpus.py \
  --dataset fineweb-edu --max-tokens 600_000_000 --output data/fineweb_edu_600m.bin
python3 lambda/run.py exec --extra cuda <id> -- \
  python examples/train_tiny_spikegpt.py --device cuda --vocab bpe \
  --train-bin data/fineweb_edu_600m.bin --layers 4 --embedding 128 ...
```

### ⚠️ You are billed by the hour until you terminate

- `python3 lambda/run.py ls` — everything still running (and charging).
- `python3 lambda/run.py terminate --all` — the panic button.
- One-shot `smoke`/`run`/`train` self-terminate even on error or Ctrl-C; a
  `launch`ed instance stays up until *you* terminate it.

The rented H100 (sm_90) is a different GPU architecture than the RTX 5090
(sm_120), so use it for A/B experiments, convergence curves, and memory/capacity
checks — not for headline wall-clock numbers. Full command list:
[`lambda/README.md`](lambda/README.md).

### Troubleshooting

- **`error: No capacity for gpu_1x_h100_sxm5 anywhere right now`**: two things at
  once. First, `gpu_1x_h100_sxm5` is the *default* type — if you passed `--type`
  and still see the default here, the flag landed *after* `--` and was ignored;
  move it before the separator. Second, that type genuinely has no capacity right
  now: run `python3 lambda/run.py types` and pass a `--type` that shows an
  available region (e.g. `--type gpu_1x_a100_sxm4`).

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
