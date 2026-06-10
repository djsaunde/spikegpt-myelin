#!/usr/bin/env bash
# Watch for Phase 3 pretrain completion, snapshot the final pretrain checkpoint
# (for later play), then auto-launch Phase 4 (WikiText fine-tune + eval).
#
# Detached (nohup) so it survives the session. Triggers off the run_with_resume
# wrapper exiting AND the checkpoint having actually reached the target step --
# so it does NOT fire Phase 4 if the wrapper died early (e.g. a fatal error),
# only on genuine completion. Phase 4 needs the GPU, which is free once the
# pretrain process has exited.
set -u
cd "$(dirname "$0")/.."

TARGET=610000
CKPT=runs/spikegpt_216m_owt2_10b.pt
BEST=runs/spikegpt_216m_owt2_10b.best.pt
LOG=runs/phase4_autostart.log

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

echo "[$(ts)] autostart: watching run_with_resume wrapper (target $TARGET steps)" >> "$LOG"
while pgrep -f "run_with_resume.sh" >/dev/null; do sleep 120; done

steps="$(completed_steps)"
echo "[$(ts)] wrapper exited; checkpoint at step ${steps}" >> "$LOG"
if [ "${steps:-0}" -lt "$TARGET" ]; then
  echo "[$(ts)] ABORT: pretrain did not reach $TARGET (got ${steps}). NOT launching Phase 4 — investigate." >> "$LOG"
  exit 1
fi

# Preserve the final + best pretrain checkpoints (Phase 4 writes to other names,
# but snapshot anyway so they are safe to play with independently).
cp -v "$CKPT" runs/spikegpt_216m_owt2_10b.final.pt >> "$LOG" 2>&1
cp -v "$BEST" runs/spikegpt_216m_owt2_10b.best.preserved.pt >> "$LOG" 2>&1
echo "[$(ts)] preserved final+best checkpoints; launching Phase 4" >> "$LOG"

PRETRAINED="$BEST" bash examples/phase4_finetune_eval.sh >> runs/phase4_run.log 2>&1
echo "[$(ts)] Phase 4 finished (exit $?). See runs/phase4_run.log" >> "$LOG"
