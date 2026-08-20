#!/usr/bin/env bash
# Unity environment setup for spikegpt-myelin. Run ONCE on a compute node
# (via the smoke test's `salloc`, NOT the login node -- the login node has no GPU
# and building torch wheels there wastes shared login resources).
#
#   salloc -p gpu --gres=gpu:a100:1 -c 8 --mem=48G -t 01:00:00
#   cd $HOME/spikegpt-myelin && bash unity/setup_env.sh
#
# What it does: installs uv (user-local, no root), syncs the project env from
# uv.lock (the SAME cu130 nightly pins we use on the 5090 -- cu130 wheels support
# Ampere sm_80 / Hopper sm_90 too), then prints a GPU/driver/torch diagnostic so
# we can confirm the nightly runs on Unity's driver before committing to a hero run.
set -euo pipefail

REPO="${REPO:-$HOME/spikegpt-myelin}"
cd "$REPO"

echo "==================== Unity env setup ===================="
echo "host: $(hostname)   repo: $REPO"

# --- 1. CUDA driver check (the ONE real risk with the cu130 nightly) ----------
# cu130 wheels bundle the CUDA 13 runtime but still need a driver >= ~580.
# If `nvidia-smi` shows an older driver, we fall back to a cu12x nightly below.
echo; echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || {
  echo "!! no GPU visible -- are you on a compute node with --gres=gpu:...?"; exit 1; }

# --- 2. uv (user-local) -------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo; echo "--- installing uv (user-local) ---"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# --- 3. GitHub auth for the private myelin git dep ----------------------------
# uv.lock pins myelin from github.com/djsaunde/myelin.git. If that repo is private
# on the cluster, export a token first:  export GIT_TOKEN=ghp_xxx
# (we rewrite https auth just for this sync; nothing is written to disk).
if [ -n "${GIT_TOKEN:-}" ]; then
  git config --global url."https://x-access-token:${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"
  echo "git: token auth configured for github.com"
fi

# --- 4. sync the locked env ---------------------------------------------------
echo; echo "--- uv sync (--extra cuda --extra tracking --extra tokenization) ---"
# --frozen: install exactly uv.lock; do NOT re-resolve (keeps the 5090-validated pins).
if ! uv sync --frozen --extra cuda --extra tracking --extra tokenization; then
  echo
  echo "!! cu130 sync failed. Most likely the Unity driver is < the cu130 minimum."
  echo "   Fallback: retry against an older CUDA nightly by editing pyproject's"
  echo "   [[tool.uv.index]] url to .../nightly/cu128 (or cu126) and re-running"
  echo "   'uv sync --extra cuda --extra tracking --extra tokenization' (drops --frozen so it re-resolves)."
  echo "   Check the required driver against 'nvidia-smi' output above."
  exit 1
fi

# --- 5. diagnostic: does torch actually see the GPU? --------------------------
echo; echo "--- torch / GPU diagnostic ---"
uv run --extra cuda --extra tracking --extra tokenization python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device:", torch.cuda.get_device_name(i))
    cap = torch.cuda.get_device_capability(i)
    print("compute capability: sm_%d%d" % cap)          # a100=sm_80, h200=sm_90
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    # tiny matmul to confirm the runtime actually executes on-device
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    print("on-device matmul: OK")
PY

echo; echo "==================== setup complete ===================="
echo "Next: sbatch unity/smoke_test.sbatch   (validates a real training step)"
