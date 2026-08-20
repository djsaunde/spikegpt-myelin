#!/usr/bin/env bash
#SBATCH --job-name=spikegpt-smoke
#SBATCH --partition=gpu                        # confirmed: 14-day walltime, has a100/h100
#SBATCH --account=pi_mhajiesmaili_umass_edu    # (also the default account)
#SBATCH --qos=short                            # 4h cap, higher priority -- ideal for a smoke test
#SBATCH --gres=gpu:a100:1                      # 1x A100-80GB; h100:1 or h200_nvl:1 also fit
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=unity/logs/smoke_%j.out
#SBATCH --error=unity/logs/smoke_%j.out
#
# End-to-end validation on Unity hardware: env -> data -> compile -> WKV autograd ->
# bf16 training step. Self-contained: it synthesizes a small random-token corpus so
# it does NOT depend on staging the FineWeb corpus first. A clean run means the
# 5090-validated cu130 nightly + our Triton WKV kernel work on Unity's A100/H200.
#
#   sbatch unity/smoke_test.sbatch
#   tail -f unity/logs/smoke_<jobid>.out
set -euo pipefail

REPO="${REPO:-$HOME/spikegpt-myelin}"
cd "$REPO"
mkdir -p unity/logs
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==================== spikegpt smoke test ===================="
echo "job $SLURM_JOB_ID on $(hostname) @ $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- synth corpus (20M uint16 tokens, ~40MB) if none present ------------------
SMOKE_BIN="$REPO/unity/smoke_corpus.bin"
if [ ! -f "$SMOKE_BIN" ]; then
  echo "--- synthesizing smoke corpus at $SMOKE_BIN ---"
  uv run python - "$SMOKE_BIN" <<'PY'
import sys, json, numpy as np
from pathlib import Path
p = Path(sys.argv[1]); n, vocab = 20_000_000, 50277
# reproducible without Math.random-style nondeterminism: fixed seed
rng = np.random.default_rng(0)
rng.integers(0, vocab, size=n, dtype=np.uint16).tofile(p)
p.with_suffix(".json").write_text(json.dumps({"n_tokens": n, "vocab_size": vocab, "dtype": "uint16"}))
print("wrote", p, n, "tokens")
PY
fi

# --- 200-step training smoke: d512/L12, ctx1024, batch 8, compile regional -----
# batch 8 (not 16) keeps it well under memory on any GPU tier; this is a plumbing
# check, not a throughput benchmark. Watch the per-step time in the log.
echo "--- training 200 steps (d512 L12, bf16, compile regional) ---"
uv run --extra cuda --extra tracking --extra tokenization python examples/train_tiny_spikegpt.py \
  --device cuda --compile regional --compile-mode default --matmul-precision high --amp bf16 \
  --train-bin "$SMOKE_BIN" --vocab bpe \
  --val-holdout-tokens 1000000 --val-eval strided --val-eval-tokens 500000 \
  --context-length 1024 --layers 12 --embedding 512 --model-type rwkv \
  --batch 8 --steps 200 \
  --lr 0.0009 --lr-final 9e-05 --lr-schedule cosine --warmup-steps 20 \
  --weight-decay 0.1 --dropout 0.0 --grad-clip 1.0 \
  --log-every 25 --eval-every 100 --sample-tokens 0

echo "==================== smoke test OK @ $(date -u +%FT%TZ) ===================="
echo "If loss decreased and per-step time is sane, the env is ready for a hero run."
