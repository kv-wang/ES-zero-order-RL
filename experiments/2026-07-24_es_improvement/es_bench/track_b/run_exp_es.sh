#!/usr/bin/env bash
# args: GPU METHOD(fulles|loraes) N1 [N2 ...]
# env: MODEL, DATASET, OUTDIR, TAGPRE, STEPS. Records wall-clock + peak-VRAM (nvidia-smi sampler).
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
GPU=$1; METHOD=$2; shift 2
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}; DATASET=${DATASET:-gsm8k}
OUT=${OUTDIR:-track_b/results/exp1}; LOG=${OUTDIR:-track_b/logs/exp1}; TAGPRE=${TAGPRE:-${METHOD}}; STEPS=${STEPS:-200}
OUT=${OUTDIR:-track_b/results/exp1}; LOG=track_b/logs/$(basename $OUT); mkdir -p "$OUT" "$LOG"
for N in "$@"; do
  tag="${TAGPRE}_N${N}"; VRAM="$OUT/${tag}_vram.txt"
  echo "[$(date +%T)] GPU$GPU START $tag ($MODEL $DATASET)" >> "$LOG/driver.log"
  ( m=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null|head -1); if [ "${u:-0}" -gt "$m" ] 2>/dev/null; then m=$u; echo $m > "$VRAM"; fi; sleep 3 || true; done ) & SAMP=$!
  t0=$(date +%s)
  if [ "$METHOD" = "fulles" ]; then
    $P4_PY track_b/src/es_train_fullparam.py --reward binary --model "$MODEL" --dataset "$DATASET" --population_size $N \
      --num_steps $STEPS --sigma 1e-3 --alpha 5e-4 --pop_seed 0 --gpu $GPU --eval_final --eval_cap 300 --kl \
      --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  else
    $P4_PY track_a/src/es_train_lora.py --model "$MODEL" --dataset "$DATASET" --population_size $N --batch 8 --num_steps $STEPS \
      --sigma 0.005 --alpha 2e-3 --pop_seed 0 --gpu $GPU --eval_every 100 --eval_cap 200 \
      --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  fi
  rc=$?; kill $SAMP 2>/dev/null
  echo "[$(date +%T)] GPU$GPU DONE $tag rc=$rc wall=$(( $(date +%s)-t0 ))s peakVRAM=$(cat $VRAM 2>/dev/null)MiB" >> "$LOG/driver.log"
done
