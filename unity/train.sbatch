#!/usr/bin/env bash
#SBATCH --job-name=spikegpt-hero
#SBATCH --partition=gpu                        # confirmed: 14-day walltime, has a100/h100
#SBATCH --account=pi_mhajiesmaili_umass_edu    # (also the default account)
#SBATCH --qos=long                             # 14-day cap -- the whole hero fits in ONE job
#SBATCH --gres=gpu:h100:1                      # h100:1 (fastest here); a100:1 also fits 1.6B bf16.
                                               # H200: --partition=gpupod-h200 --gres=gpu:h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00         # single job covers the run; self-resubmit below is insurance
#SBATCH --output=unity/logs/hero_%x_%j.out
#SBATCH --error=unity/logs/hero_%x_%j.out
#
# Walltime-spanning single-GPU pretraining with automatic checkpoint/resume.
# The 1.6B hero exceeds a single SLURM walltime block (~2.1 d on H200, ~6.6 d on
# A100), so each job:
#   1. reads previous_steps from the checkpoint (0 on a fresh start),
#   2. submits its SUCCESSOR up front with `afterany` on itself, so the chain
#      survives even a hard walltime/OOM kill (not just a clean exit),
#   3. trains `TARGET_STEPS - previous_steps` more steps -- passing `remaining`
#      keeps total_steps = previous_steps + remaining = TARGET_STEPS, so the cosine
#      horizon stays pinned across every resume (see trainer L582),
#   4. checkpoints every CKPT_EVERY steps to CKPT so the successor resumes cleanly.
# When previous_steps >= TARGET_STEPS the job exits immediately (cheap no-op tail).
#
# Config is via env vars (all have defaults for the planned ~1.6B hero). Example:
#   RUN_NAME=hero_1p6b sbatch unity/train.sbatch
#   # A100 instead:  sbatch --gres=gpu:a100:1 --time=2-00:00:00 unity/train.sbatch
#
# DO NOT launch a large paid/long run without Dan's explicit approval.
set -euo pipefail

REPO="${REPO:-$HOME/spikegpt-myelin}"
cd "$REPO"
mkdir -p unity/logs unity/ckpt
export PATH="$HOME/.local/bin:$PATH"
export WANDB_ENTITY="${WANDB_ENTITY:-pitheta}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- hero config (override via env) ------------------------------------------
RUN_NAME="${RUN_NAME:-hero_1p6b}"
LAYERS="${LAYERS:-32}"
EMBED="${EMBED:-1792}"
LR="${LR:-9e-05}"                 # set from the lr(d) rule for this width before launch
LR_FINAL="${LR_FINAL:-9e-06}"     # cosine to lr/10
TARGET_STEPS="${TARGET_STEPS:-0}" # REQUIRED: set from the token budget (tokens/(batch*ctx))
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"     # raise if the batch OOMs on the chosen GPU
CTX="${CTX:-1024}"
WARMUP="${WARMUP:-2000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
# corpus + checkpoints live on /work (1.1 PB), NOT $HOME (100 GB) -- the corpus is ~50 GB.
WORK="${WORK:-/work/pi_mhajiesmaili_umass_edu/csigrist_spikegpt}"
CORPUS="${CORPUS:-$WORK/data/fineweb-edu_sample-100BT_25000000000.bin}"
VAL_HOLDOUT="${VAL_HOLDOUT:-50000000}"
WANDB_PROJECT="${WANDB_PROJECT:-spikegpt-scaling}"
EXTRA_ARGS="${EXTRA_ARGS:-}"      # e.g. "--no-spiking" for a dense baseline
mkdir -p "$WORK/ckpt"
CKPT="$WORK/ckpt/${RUN_NAME}.ckpt"

if [ "$TARGET_STEPS" -le 0 ]; then
  echo "!! TARGET_STEPS unset. Set it from the token budget: tokens/(BATCH*GRAD_ACCUM*CTX)."; exit 2
fi
[ -f "$CORPUS" ] || { echo "!! corpus missing: $CORPUS (stage it first, see unity/README.md)"; exit 2; }

echo "==================== $RUN_NAME  job $SLURM_JOB_ID ===================="
echo "$(hostname) @ $(date -u +%FT%TZ)   d=$EMBED L=$LAYERS target=$TARGET_STEPS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# ---- how many steps already done? --------------------------------------------
done_steps() {
  uv run python - "$CKPT" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(ck.get("metadata", {}).get("previous_steps", 0)))
except Exception:
    print(0)
PY
}
DONE="$(done_steps)"
echo "previous_steps=$DONE / target=$TARGET_STEPS"
if [ "$DONE" -ge "$TARGET_STEPS" ]; then
  echo "target already reached -- nothing to do. Done."; exit 0
fi
REMAINING=$((TARGET_STEPS - DONE))

# ---- submit successor FIRST (survives a hard kill of this job) ----------------
# afterany: run the successor whether we finish, hit walltime, or crash. The
# successor re-checks done_steps and no-ops if we happened to reach target.
if command -v sbatch >/dev/null 2>&1; then
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
           --job-name="${SLURM_JOB_NAME}" --partition="${SLURM_JOB_PARTITION}" \
           --export=ALL unity/train.sbatch || true)
  [ -n "${NEXT:-}" ] && echo "queued successor job $NEXT (afterany:${SLURM_JOB_ID})"
fi

RESUME=(); [ -f "$CKPT" ] && RESUME=(--checkpoint-in "$CKPT")
echo "--- training $REMAINING steps${RESUME:+ (resuming from $CKPT)} ---"
# shellcheck disable=SC2086
uv run --extra cuda --extra tracking python examples/train_tiny_spikegpt.py \
  --device cuda --compile regional --compile-mode max-autotune --matmul-precision high --amp bf16 \
  --train-bin "$CORPUS" --vocab bpe \
  --val-holdout-tokens "$VAL_HOLDOUT" --val-eval strided --val-eval-tokens 2000000 \
  --context-length "$CTX" --layers "$LAYERS" --embedding "$EMBED" --model-type rwkv \
  --batch "$BATCH" --grad-accum "$GRAD_ACCUM" --steps "$REMAINING" "${RESUME[@]}" \
  --lr "$LR" --lr-final "$LR_FINAL" --lr-schedule cosine --warmup-steps "$WARMUP" \
  --weight-decay 0.1 --dropout 0.0 --grad-clip 1.0 \
  --checkpoint-out "$CKPT" --checkpoint-every "$CKPT_EVERY" --best-checkpoint-out "${CKPT%.ckpt}.best.pt" \
  --log-every 100 --eval-every 2500 --sample-tokens 0 \
  --wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$RUN_NAME" \
  $EXTRA_ARGS

echo "==================== chunk done @ $(date -u +%FT%TZ), previous_steps=$(done_steps) ===================="
