#!/usr/bin/env bash
# Canonical hero-run config + helpers. Sourced by watchdog.sh and usable by hand
# to (re)launch or check the hero. Single source of truth for the launch params so
# an automatic resubmit is byte-identical to the original launch.
HERO_RUN_NAME=hero_c1e20_d2048
HERO_WORK=/work/pi_mhajiesmaili_umass_edu/csigrist_spikegpt
HERO_CKPT="$HERO_WORK/ckpt/${HERO_RUN_NAME}.ckpt"
HERO_DONE="$HERO_WORK/${HERO_RUN_NAME}.DONE"
HERO_REPO="${HOME}/spikegpt-myelin"

# submit the hero (resumes from checkpoint automatically via done_steps in train.sbatch)
hero_submit() {
  cd "$HERO_REPO" || return 1
  RUN_NAME=$HERO_RUN_NAME \
  LAYERS=41 EMBED=2048 \
  LR=6.454e-05 LR_FINAL=6.454e-06 WARMUP=2000 \
  BATCH=16 GRAD_ACCUM=4 \
  TARGET_STEPS=434996 \
  sbatch --parsable --gres=gpu:h100:1 --partition=gpu,gpu-preempt unity/train.sbatch
}

# count of live hero jobs (running OR pending) for this user
hero_alive_count() {
  squeue --me -h -o "%j %T" 2>/dev/null | awk '$1=="spikegpt-hero"' | wc -l | tr -d ' '
}

# has the run completed? (marker written by train.sbatch when done_steps>=target)
hero_is_done() { [ -f "$HERO_DONE" ]; }
