#!/usr/bin/env bash
# LR-rule pilot, all-local RTX 5090, batch 16 (d384 + d512 + d768).
#
# Phase-1 of the scaling study fits lr_opt(d) = a * d^b from a few short runs per
# width, so the main grid can set a principled LR at each width. The whole study
# runs on this 32GB 5090 at batch 16 -- batch 32 oversubscribes VRAM here (d768
# thrashes at ~12s/step; batch 16 fits at 31.6GB/97%, 196ms/step). The 3 earlier
# d384 runs were batch 32 (on a rented H100) and are OFF-REGIME, so d384 is
# re-run here at batch 16 to keep all three widths at one batch.
#
# Each run: 750M tokens single-epoch, ctx 1024, cosine to lr/10, wd 0.1, bf16,
# regional/default compile. The 3 LRs per width bracket the batch-16 1/d
# prediction (batch-16 d384 optimum estimated ~1.4e-3 = 2.01e-3 * sqrt(1/2));
# fit_lr_rule.py flags + suggests an extension if a width's optimum lands on a
# bracket endpoint (extend that width by one ~2.5x step and re-run).
#
# Widths ordered d384 -> d512 -> d768 (roomiest first): d768 is the tight one, so
# if it hits memory trouble the two smaller widths (enough for a fit) are done.
#
# Usage:
#   examples/scaling/pilot_local.sh probe   # 400-step d768 capacity/throughput check
#   examples/scaling/pilot_local.sh plan    # print the 9 runs + derived steps/warmup, launch nothing
#   examples/scaling/pilot_local.sh run     # run all 9 arms sequentially (overnight, ~15h)
set -euo pipefail

cd "$(dirname "$0")/../.."

# --- Config -----------------------------------------------------------------
BATCH="${BATCH:-16}"                 # per-step batch; 16 is the 5090 ceiling at these widths
TOKENS="${TOKENS:-750000000}"        # single-epoch token budget per run
CTX=1024
WD=0.1
CORPUS="${CORPUS:-data/fineweb-edu_sample-100BT_800000000.bin}"
VAL_HOLDOUT=5000000                  # tail of the corpus held out as in-domain val
VAL_EVAL_TOKENS=2000000              # cap in-loop strided eval to first 2M val tokens (cost guard)
WANDB_PROJECT="${WANDB_PROJECT:-spikegpt-scaling}"
export WANDB_ENTITY="${WANDB_ENTITY:-pitheta}"   # runs land in pitheta/<project>; NOT the default local login
CKPT="${CKPT:-0}"                     # 1 => --activation-checkpointing (7x slower; last resort for OOM)
# expandable_segments cuts allocator fragmentation -- headroom matters, d768 peaks at 97% VRAM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Arms: "d L lr" -- 3 LRs per width, bracketing the batch-16 1/d prediction from
# an estimated batch-16 d384 optimum ~1.4e-3 (2.01e-3 batch-32 vertex * sqrt(1/2)).
#   d384 predict ~1.4e-3  -> bracket 0.8 / 1.4 / 2.5 e-3  (also measures the real b16 anchor)
#   d512 predict ~1.05e-3 -> bracket 0.6 / 1.1 / 2.0 e-3
#   d768 predict ~0.70e-3 -> bracket 0.4 / 0.75 / 1.4 e-3
ARMS=(
  "384 10 0.0008"
  "384 10 0.0014"
  "384 10 0.0025"
  "512 12 0.0006"
  "512 12 0.0011"
  "512 12 0.0020"
  "768 18 0.0004"
  "768 18 0.00075"
  "768 18 0.0014"
)
# Override the arm list for a targeted extension, e.g. re-bracketing a width whose
# optimum fit_lr_rule.py flagged as out-of-bracket. Semicolon-separated "d L lr":
#   ARMS_LIST="768 18 0.00016;768 18 0.0003" examples/scaling/pilot_local.sh run
if [ -n "${ARMS_LIST:-}" ]; then IFS=';' read -ra ARMS <<<"$ARMS_LIST"; fi

steps_for() { python3 -c "import math,sys; print(math.ceil($TOKENS/(int(sys.argv[1])*$CTX)))" "$1"; }
warmup_for() { python3 -c "import sys; s=int(sys.argv[1]); print(max(100,min(2000,s//20)))" "$1"; }
lrfinal_for() { python3 -c "import sys; print(f'{float(sys.argv[1])/10:g}')" "$1"; }

run_one() {
  local d="$1" L="$2" lr="$3" name="$4" steps warmup lrf extra=()
  steps="$(steps_for "$BATCH")"
  warmup="$(warmup_for "$steps")"
  lrf="$(lrfinal_for "$lr")"
  [ "$CKPT" = "1" ] && extra+=(--activation-checkpointing)
  echo ">>> $name  d=$d L=$L lr=$lr lr_final=$lrf batch=$BATCH steps=$steps warmup=$warmup"
  [ "${PLAN:-0}" = "1" ] && return 0
  # --compile-mode default: skip the ~35min max-autotune; probe showed 196ms/step at d768.
  uv run --extra cuda --extra tracking python examples/train_tiny_spikegpt.py \
    --device cuda --compile regional --compile-mode default --matmul-precision high --amp bf16 \
    --train-bin "$CORPUS" --vocab bpe \
    --val-holdout-tokens "$VAL_HOLDOUT" \
    --val-eval strided --val-eval-tokens "$VAL_EVAL_TOKENS" \
    --context-length "$CTX" --layers "$L" --embedding "$d" --model-type rwkv \
    --batch "$BATCH" --steps "$steps" \
    --lr "$lr" --lr-final "$lrf" --lr-schedule cosine --warmup-steps "$warmup" \
    --weight-decay "$WD" --dropout 0.0 --grad-clip 1.0 \
    --log-every 100 --eval-every 1000 --sample-tokens 16 \
    --wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$name" \
    "${extra[@]}" 2>&1 | tee "runs/logs/${name}.log"
}

cmd="${1:-plan}"
case "$cmd" in
  probe)
    # Worst-case memory/throughput check: d768 at the chosen BATCH, 400 steps,
    # no eval, no wandb. Watch `nvidia-smi` in another shell for peak VRAM.
    steps=400
    echo ">>> PROBE d768 batch=$BATCH steps=$steps (compile regional, no eval/wandb)"
    extra=(); [ "$CKPT" = "1" ] && extra+=(--activation-checkpointing)
    uv run --extra cuda --extra tracking python examples/train_tiny_spikegpt.py \
      --device cuda --compile regional --matmul-precision high --amp bf16 \
      --train-bin "$CORPUS" --vocab bpe --val-holdout-tokens "$VAL_HOLDOUT" \
      --context-length "$CTX" --layers 18 --embedding 768 --model-type rwkv \
      --batch "$BATCH" --steps "$steps" --lr 0.0012 --warmup-steps 100 \
      --weight-decay "$WD" --dropout 0.0 --grad-clip 1.0 \
      --log-every 25 --eval-every 100000 --sample-tokens 0 "${extra[@]}"
    ;;
  plan)
    PLAN=1
    for arm in "${ARMS[@]}"; do read -r d L lr <<<"$arm"; run_one "$d" "$L" "$lr" "pilot_d${d}_lr${lr}"; done
    echo "(plan only: set '$0 run' to launch)"
    ;;
  run)
    [ -f "$CORPUS" ] || { echo "corpus missing: $CORPUS (build it with prepare_token_corpus.py)"; exit 1; }
    failed=()
    for arm in "${ARMS[@]}"; do
      read -r d L lr <<<"$arm"
      name="pilot_d${d}_lr${lr}"
      # Tolerate a single run dying (e.g. d768 OOM) without aborting the fleet.
      run_one "$d" "$L" "$lr" "$name" || { echo "!!! run $name FAILED (rc=$?), continuing"; failed+=("$name"); }
    done
    echo "pilot done. failed=${failed[*]:-none} -> analyze with pilot_results.py + fit_lr_rule.py"
    ;;
  *) echo "usage: $0 {probe|plan|run}"; exit 2 ;;
esac
