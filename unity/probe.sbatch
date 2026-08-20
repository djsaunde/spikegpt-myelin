#!/usr/bin/env bash
#SBATCH --job-name=spikegpt-probe
#SBATCH --partition=gpu
#SBATCH --account=pi_mhajiesmaili_umass_edu
#SBATCH --qos=short                # 4h cap, high priority
#SBATCH --gres=gpu:h100:1          # probe on the SAME GPU the hero will use
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=unity/logs/probe_%j.out
#SBATCH --error=unity/logs/probe_%j.out
#
# Throughput + VRAM probe for the hero architecture. Runs the REAL d2048/L41 model
# for a short burst per config (no eval, no wandb), sampling peak GPU memory and
# reporting steady-state ms/step + tokens/s. Science-fixed knobs (effective batch,
# ctx, arch, lr) are held; only speed/memory knobs vary.
#
# Sweep is CONFIGS: semicolon-separated "grad_accum,compile,compile_mode".
#   CONFIGS="1,regional,default;2,regional,default;4,regional,default" sbatch unity/probe.sbatch
set -uo pipefail

REPO="${REPO:-$HOME/spikegpt-myelin}"; cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p unity/logs

EMBED="${EMBED:-2048}"; LAYERS="${LAYERS:-41}"; BATCH="${BATCH:-16}"; CTX="${CTX:-1024}"
STEPS="${STEPS:-120}"
CORPUS="${CORPUS:-/work/pi_mhajiesmaili_umass_edu/csigrist_spikegpt/data/fineweb-edu_sample-100BT_25000000000.bin}"
CONFIGS="${CONFIGS:-1,regional,default;2,regional,default;4,regional,default}"

echo "==================== hero throughput probe ===================="
echo "$(hostname) @ $(date -u +%FT%TZ)   d=$EMBED L=$LAYERS batch=$BATCH ctx=$CTX steps=$STEPS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
[ -f "$CORPUS" ] || { echo "!! corpus missing: $CORPUS"; exit 2; }

peak_sampler() { # $1=outfile ; samples used-MiB every 0.5s until killed
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$1" 2>/dev/null; sleep 0.5; done
}

printf "\n%-8s %-9s %-13s %-11s %-12s %-10s\n" accum compile mode peakVRAM steady_ms tokens/s
echo "-------------------------------------------------------------------------"
IFS=';' read -ra CFGS <<<"$CONFIGS"
for cfg in "${CFGS[@]}"; do
  IFS=',' read -r GA CMP CMODE <<<"$cfg"
  SMP="unity/logs/.probe_mem_$$"; : >"$SMP"
  peak_sampler "$SMP" & SPID=$!
  LOG="unity/logs/.probe_run_${SLURM_JOB_ID}_${GA}_${CMP}_${CMODE}.log"
  # micro-batch = BATCH/GA must be integer
  uv run --extra cuda --extra tracking --extra tokenization python examples/train_tiny_spikegpt.py \
    --device cuda --compile "$CMP" --compile-mode "$CMODE" --matmul-precision high --amp bf16 \
    --train-bin "$CORPUS" --vocab bpe --val-holdout-tokens 50000000 \
    --context-length "$CTX" --layers "$LAYERS" --embedding "$EMBED" --model-type rwkv \
    --batch "$BATCH" --grad-accum "$GA" --steps "$STEPS" \
    --lr 6.454e-05 --lr-final 6.454e-06 --lr-schedule cosine --warmup-steps 20 \
    --weight-decay 0.1 --dropout 0.0 --grad-clip 1.0 \
    --log-every 40 --eval-every 100000 --sample-tokens 0 >"$LOG" 2>&1
  RC=$?
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  PEAK=$(sort -n "$SMP" | tail -1); rm -f "$SMP"
  if [ $RC -ne 0 ]; then
    if grep -qiE "out of memory|CUDA out of memory" "$LOG"; then STAT="OOM"; else STAT="ERR(rc=$RC)"; fi
    printf "%-8s %-9s %-13s %-11s %-12s %-10s\n" "$GA" "$CMP" "$CMODE" "${PEAK:-?}MiB" "$STAT" "-"
    grep -iE "out of memory|Error|Traceback" "$LOG" | head -2 | sed 's/^/    /'
    continue
  fi
  MS=$(grep -oE "steady_state_average_step_ms=[0-9.]+" "$LOG" | head -1 | cut -d= -f2)
  [ -z "$MS" ] && MS=$(grep -oE "post_warmup_average_step_ms=[0-9.]+" "$LOG" | head -1 | cut -d= -f2)
  TPS=$(awk -v b="$BATCH" -v c="$CTX" -v ms="$MS" 'BEGIN{ if(ms>0) printf "%.0f", b*c*1000/ms; else print "?"}')
  printf "%-8s %-9s %-13s %-11s %-12s %-10s\n" "$GA(mb$((BATCH/GA)))" "$CMP" "$CMODE" "${PEAK}MiB" "${MS}ms" "$TPS"
done
echo "-------------------------------------------------------------------------"
echo "done @ $(date -u +%FT%TZ). Higher tokens/s = better; keep peakVRAM under ~76000MiB for headroom."
