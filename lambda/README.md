# Lambda Cloud harness

A VM-based counterpart to `modal/` for running GPU experiments on
[Lambda Cloud](https://cloud.lambda.ai/). Owns the whole lifecycle — **launch →
wait for SSH → provision our frozen env → rsync repo + data up → run → fetch →
terminate** — and *always* terminates anything it launched, even on error/Ctrl-C,
so a failed run can't burn credits. The driver (`harness.py` + `run.py`) is
stdlib-only; run it with the system `python3`.

Step-by-step setup and examples (for someone new): see the **"Running on rented
GPUs"** section of [`../ONBOARDING.md`](../ONBOARDING.md). Quick reference below.

## Setup

```bash
export LAMBDA_API_KEY=secret_...        # https://cloud.lambda.ai/api-keys
python3 lambda/run.py keys --add        # register ~/.ssh/id_ed25519.pub (once; --key to override)
```

Provisioning clones the private `myelin` dep using a GitHub token from
`GH_TOKEN`/`GITHUB_TOKEN` or `gh auth token`, and installs NVIDIA
`cuda-compat-13-0` so the pinned cu130 torch runs on Lambda's (CUDA-12.8) driver —
keeping the frozen env identical to local, no relock.

## Commands

| Command | What it does |
|---|---|
| `types [--gpu S]` | Instance types, price, regions with capacity |
| `keys [--add]` | List SSH keys; `--add` registers your local pubkey |
| `ls` | List running instances |
| `launch` | Launch a persistent instance; prints `<id> <ip>` |
| `provision <id\|ip>` | rsync repo + `uv sync --frozen --extra cuda` |
| `exec <id\|ip> -- CMD` | Run CMD in the synced uv env |
| `fetch <id\|ip> REMOTE LOCAL` | rsync an artifact down (REMOTE relative to repo root) |
| `terminate <id...> \| --all` | Terminate instances (stops billing) |
| `smoke` | One-shot GPU/env smoke test |
| `run -- CMD` | One-shot: launch → provision → CMD → fetch `runs/` → terminate |
| `train [flags] -- ...` | One-shot enwik8 training run |

One-shot `smoke`/`run`/`train` self-terminate on exit unless given `--keep` or
`--instance-id <id>`. Shared flags: `--type` (default `gpu_1x_h100_sxm5`, sm_90 —
use for A/Bs and convergence, not headline wall-clock vs the sm_120 local GPU),
`--region`, `--name`, `--key`, `--extra`, `--keep`, `--instance-id`,
`--provision`, `--no-fetch`.

> ⚠️ You're billed hourly until termination. `ls` shows what's running;
> `terminate --all` is the panic button.
