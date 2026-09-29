#!/usr/bin/env bash
# args: GPU METHOD(fulles|loraes) N1 [N2 ...]   -> MATH L3-5, 1.5B-Instruct, 200 steps, seed0
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
GPU=$1; METHOD=$2; shift 2
OUT=track_b/results/exp1; LOG=track_b/logs/exp1; mkdir -p "$OUT" "$LOG"
for N in "$@"; do
  tag="${METHOD}_math_N${N}"
  echo "[$(date +%T)] GPU$GPU START $tag" >> "$LOG/driver.log"
  if [ "$METHOD" = "fulles" ]; then
    $P4_PY track_b/src/es_train_fullparam.py --reward binary --dataset math --population_size $N \
      --num_steps 200 --sigma 1e-3 --alpha 5e-4 --pop_seed 0 --gpu $GPU --eval_final --eval_cap 300 --kl \
      --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  else
    $P4_PY track_a/src/es_train_lora.py --dataset math --population_size $N --batch 8 --num_steps 200 \
      --sigma 0.005 --alpha 2e-3 --pop_seed 0 --gpu $GPU --eval_every 100 --eval_cap 200 \
      --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  fi
  echo "[$(date +%T)] GPU$GPU DONE $tag rc=$?" >> "$LOG/driver.log"
done
echo "[$(date +%T)] GPU$GPU lane done ($METHOD)" >> "$LOG/driver.log"
