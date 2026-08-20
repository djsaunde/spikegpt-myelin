#!/usr/bin/env bash
# Durable cluster-side watchdog for the hero run. Runs from scrontab (SLURM cron)
# independent of any interactive session. It is a BACKSTOP: the primary recovery is
# train.sbatch's afterany self-resubmit chain. This catches the rare case where the
# whole chain is gone (e.g. a job killed before it submitted its successor).
#
# Logic (idempotent, no-duplicate): if the run is DONE -> nothing. Else if any hero
# job is alive (running or pending) -> heartbeat only. Else (chain fully dead) ->
# resubmit; it resumes from the /work checkpoint via done_steps, so no lost work and
# no double-training. Only resubmits when NOTHING is alive => cannot create duplicates.
set -uo pipefail
source "${HOME}/spikegpt-myelin/unity/hero_env.sh"
LOG="$HERO_WORK/watchdog.log"
ts() { date -u +%FT%TZ; }

if hero_is_done; then
  echo "$(ts) DONE marker present -> run complete, watchdog idle." >>"$LOG"; exit 0
fi

ALIVE=$(hero_alive_count)
if [ "${ALIVE:-0}" -ge 1 ]; then
  # heartbeat: report last checkpointed step if we can read it cheaply from the log
  L=$(ls -t "$HERO_REPO"/unity/logs/hero_spikegpt-hero_*.out 2>/dev/null | head -1)
  STEP=$(grep -oE '^\| +[0-9]+ +\|' "$L" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  echo "$(ts) OK alive=$ALIVE step=${STEP:-?}/434996" >>"$LOG"
  exit 0
fi

# nothing alive and not done -> the chain broke. Resubmit.
NEW=$(hero_submit)
echo "$(ts) !! chain DEAD (alive=0, not done) -> RESUBMITTED hero as job ${NEW:-FAILED}" >>"$LOG"
