# Running spikegpt-myelin on Unity (MGHPCC SLURM cluster)

Deploy-ready scaffolding for a single-GPU **hero run** on Unity, targeting the
`pi_mhajiesmaili_umass_edu` allocation. Cluster specifics below are **confirmed**
from the live cluster (2026-08-20), not guessed.

We log in through the collaborator's shared account:

| Item | Value |
|------|-------|
| Login | `ssh unity` -> `csigrist_umass_edu@unity.rc.umass.edu` (key: `~/.ssh/unity_key`) |
| Account | `pi_mhajiesmaili_umass_edu` (also the default account) |
| Partition | `gpu` -- **14-day** walltime; has A100-80GB and H100 |
| QOS | `short` (4 h, higher priority -- smoke tests); `long` (14 days -- hero runs) |
| GPU gres | `gpu:a100:1`, `gpu:h100:1`; H200 is `gpu:h200_nvl:1` on partition `gpupod-h200` |
| Repo + env | `$HOME/spikegpt-myelin` (home = 100 GB) |
| Corpus + checkpoints | `/work/pi_mhajiesmaili_umass_edu/csigrist_spikegpt/` (work = 1.1 PB, group-writable) |

> Shared account: other lab members keep their own subdirs under `/work/pi_mhajiesmaili_umass_edu/`.
> We use `csigrist_spikegpt/` there. Be a good tenant -- one GPU at a time unless agreed otherwise.

The 14-day walltime means the whole hero run fits in **one** job; `train.sbatch`'s
self-resubmit is just insurance against a node failure.

---

## 1. Clone the repo (on the login node)

```bash
ssh unity
cd $HOME
git clone https://github.com/djsaunde/spikegpt-myelin.git   # add a token if private
```

## 2. Build the environment (on a GPU node, not the login node)

```bash
salloc -p gpu -q short --gres=gpu:a100:1 -c 8 --mem=48G -t 01:00:00
cd $HOME/spikegpt-myelin
export GIT_TOKEN=ghp_xxx          # only if the myelin git dep is private
bash unity/setup_env.sh
```

`setup_env.sh` installs `uv` (user-local), runs `uv sync --frozen --extra cuda
--extra tracking` (the **same** cu130 nightly validated on the 5090 -- cu130 wheels
cover A100 sm_80 and H100 sm_90), and prints a torch/GPU diagnostic.

**The one real unknown** is Unity's NVIDIA driver vs. the cu130 minimum (~580).
The diagnostic prints the driver version; if `uv sync` fails on it, the script tells
you to repoint `pyproject.toml`'s `[[tool.uv.index]]` url to `.../nightly/cu128`
(or `cu126`) and re-sync. (We'll know the driver for sure from the first GPU job.)

## 3. Stage the corpus to /work

The trainer reads a flat `uint16` `.bin` + a `{n_tokens,vocab_size,dtype}` `.json`
sidecar (~50 GB for 25B tokens -- must go on `/work`, not `$HOME`):

```bash
# from the 5090 box:
DEST=csigrist_spikegpt/data
ssh unity 'mkdir -p /work/pi_mhajiesmaili_umass_edu/csigrist_spikegpt/data'
rsync -avP data/fineweb-edu_sample-100BT_25000000000.bin \
           data/fineweb-edu_sample-100BT_25000000000.json \
           unity:/work/pi_mhajiesmaili_umass_edu/$DEST/
```

## 4. Smoke test (self-contained, ~5 min of GPU)

```bash
sbatch unity/smoke_test.sbatch
squeue --me
tail -f unity/logs/smoke_<jobid>.out
```

Synthesizes a small random-token corpus (no data-staging dependency) and trains 200
steps at d512/L12. A clean run = env + Triton WKV kernel + bf16 + compile all work on
Unity hardware, and the log prints the **driver version** -- do this before staging
the 50 GB corpus.

## 5. Hero run (single job; auto-resumes if the node dies)

`train.sbatch` runs under `--qos=long` (14-day cap), so the hero fits in one job; it
also self-resubmits (`--dependency=afterany`) as insurance. Checkpoints + corpus live
on `/work` via the `WORK` var. Set the config from the finalized hero spec:

```bash
RUN_NAME=hero_1p6b \
LAYERS=32 EMBED=1792 \
LR=9e-05 LR_FINAL=9e-06 WARMUP=2000 \
BATCH=16 GRAD_ACCUM=1 \
TARGET_STEPS=<tokens / (BATCH*CTX)> \
sbatch unity/train.sbatch
```

`TARGET_STEPS` = token_budget / (BATCH * CTX). Set `LR` from the fitted `lr(d)` rule
for the hero width before launching. For H100 the default `--gres=gpu:h100:1` applies;
for H200 add `--partition=gpupod-h200 --gres=gpu:h200_nvl:1`; raise `GRAD_ACCUM`
(keeping `BATCH % GRAD_ACCUM == 0`) if the micro-batch OOMs.

Monitor: `squeue --me`, `tail -f unity/logs/hero_*_<jobid>.out`, W&B
(`pitheta/spikegpt-scaling`). Cancel with `scancel <jobid>` (kill the pending
successor too, or `scancel --me`).

> **Approval gate:** do not launch the hero (or any large/long run) without Dan's
> explicit go-ahead. This scaffolding is staged and ready; launching is a separate step.

## Rough single-GPU hero-run times (1.6B, 25B tokens)

| GPU        | est. wall | fits in one 14-day job? |
|------------|-----------|-------------------------|
| H100       | ~2 days   | yes                     |
| A100-80GB  | ~6.6 days | yes                     |

The smoke test's per-step time refines these.

## Files

- `setup_env.sh` -- uv install + `uv sync` + GPU/driver diagnostic (run on a GPU node)
- `smoke_test.sbatch` -- self-contained 200-step end-to-end validation (qos=short)
- `train.sbatch` -- hero-run template (qos=long, checkpoint/resume, /work storage)
- `logs/`, `ckpt/` -- created on first run
