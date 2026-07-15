#!/usr/bin/env bash
# Auto-resuming training wrapper for long local runs.
#
# Relaunches train_tiny_spikegpt.py until a target global step is reached,
# resuming from the periodic --checkpoint-out each time. Survives crashes and the
# overnight WSL/Windows-Update VM teardown (a relaunch picks up where the last
# checkpoint left off via the checkpoint's previous_steps + the trainer's resume).
#
# Usage:
#   examples/run_with_resume.sh <target_steps> <checkpoint_out> -- <train args...>
#
# The <train args...> must include --checkpoint-out <checkpoint_out> and must NOT
# include --steps or --checkpoint-in (this wrapper injects both). total_steps =
# target, so the LR-schedule horizon stays fixed across relaunches.
set -u

TARGET="$1"; CKPT="$2"; shift 2
[ "${1:-}" = "--" ] && shift
TRAIN_ARGS=("$@")

completed_steps() {
  # Print previous_steps recorded in the checkpoint metadata (0 if absent).
  uv run python - "$CKPT" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(ck.get("metadata", {}).get("previous_steps", 0)))
except Exception:
    print(0)
PY
}

attempt=0
while true; do
  done_steps="$(completed_steps)"
  if [ "$done_steps" -ge "$TARGET" ]; then
    echo "[resume] target reached: ${done_steps}/${TARGET} steps. Done."
    break
  fi
  remaining=$((TARGET - done_steps))
  resume=()
  [ -f "$CKPT" ] && resume=(--checkpoint-in "$CKPT")
  attempt=$((attempt + 1))
  echo "[resume] attempt ${attempt}: ${done_steps}/${TARGET} done, running ${remaining} more steps$([ -f "$CKPT" ] && echo " (resuming from $CKPT)")"
  uv run --extra cuda --extra tracking python examples/train_tiny_spikegpt.py "${TRAIN_ARGS[@]}" --steps "$remaining" "${resume[@]}"
  rc=$?
  echo "[resume] training exited rc=${rc}"
  if [ "$(completed_steps)" -ge "$TARGET" ]; then
    echo "[resume] target reached after attempt ${attempt}. Done."
    break
  fi
  sleep 15  # brief backoff before relaunch (e.g. after a VM teardown)
done
