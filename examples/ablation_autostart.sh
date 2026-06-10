#!/usr/bin/env bash
# Continuous-twin ablation handoff: wait for the continuous pretrain to finish,
# snapshot its checkpoint, then fine-tune it on WikiText-103 and eval — the
# apples-to-apples downstream comparison vs the spiking model's 26.47.
#
# Detached (nohup). Fires only on genuine completion (wrapper gone AND checkpoint
# at the target step), and only if the GPU is otherwise free.
set -u
cd "$(dirname "$0")/.."

TARGET=610000
CKPT=runs/spikegpt_216m_owt2_10b_continuous.pt
BEST=runs/spikegpt_216m_owt2_10b_continuous.best.pt
LOG=runs/ablation_autostart.log

ts() { date '+%Y-%m-%d %H:%M:%S'; }

completed_steps() {
  uv run python - "$CKPT" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(ck.get("metadata", {}).get("previous_steps", 0)))
except Exception:
    print(0)
PY
}

echo "[$(ts)] ablation autostart: watching continuous wrapper (target $TARGET)" >> "$LOG"
# Match the continuous run's wrapper specifically (the spiking one is long gone).
while pgrep -f "run_with_resume.sh.*owt2_10b_continuous" >/dev/null; do sleep 120; done

steps="$(completed_steps)"
echo "[$(ts)] continuous wrapper exited; checkpoint at step ${steps}" >> "$LOG"
if [ "${steps:-0}" -lt "$TARGET" ]; then
  echo "[$(ts)] ABORT: continuous pretrain did not reach $TARGET (got ${steps})." >> "$LOG"
  exit 1
fi

cp -v "$CKPT" runs/spikegpt_216m_owt2_10b_continuous.final.pt >> "$LOG" 2>&1
echo "[$(ts)] preserved continuous checkpoint; zero-shot + WT-103 fine-tune + eval" >> "$LOG"

{
  echo "=== continuous zero-shot WikiText test PPL (no fine-tune) ==="
  uv run python examples/evaluate_spikegpt_checkpoint.py "$BEST" \
    --tokens-bin data/wikitext103/test.bin --device cuda --batch 16

  echo "=== continuous WT-103 fine-tune (--reset-schedule, same recipe as spiking) ==="
  uv run python examples/train_tiny_spikegpt.py \
    --device cuda --checkpoint-in "$BEST" --reset-schedule \
    --train-bin data/wikitext103/train.bin --val-bin data/wikitext103/validation.bin \
    --val-holdout-tokens 0 --batch 16 \
    --steps 15000 --lr 1e-5 --lr-final 1e-6 --warmup-steps 200 --lr-schedule cosine \
    --dropout 0.0 --weight-decay 0.1 --grad-clip 1.0 --amp bf16 --matmul-precision high \
    --compile regional --compile-tail --compile-mode max-autotune-no-cudagraphs --compile-warmup \
    --log-every 200 --eval-every 500 --val-eval-tokens 250000 \
    --checkpoint-out runs/spikegpt_216m_continuous_wt103_ft.pt \
    --best-checkpoint-out runs/spikegpt_216m_continuous_wt103_ft.best.pt --checkpoint-every 2000 \
    --wandb --wandb-project myelin --wandb-run-name wt103_216m_continuous_ft

  echo "=== continuous WT-103 fine-tuned -> test PPL (vs spiking 26.47) ==="
  uv run python examples/evaluate_spikegpt_checkpoint.py \
    runs/spikegpt_216m_continuous_wt103_ft.best.pt \
    --tokens-bin data/wikitext103/test.bin --device cuda --batch 16
} >> runs/ablation_finetune.log 2>&1

echo "[$(ts)] ablation fine-tune + eval finished (exit $?). See runs/ablation_finetune.log" >> "$LOG"
