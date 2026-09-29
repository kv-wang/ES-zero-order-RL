#!/usr/bin/env bash
# A1 (round 2): lower sigma {0.002,0.005} + larger batch B=32 (both A1-diagnosed fixes).
# 1.5B-Instruct, GSM8K, N=16, 300 steps, 1 seed, alpha=2e-3. One sigma per GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
OUT=track_a/results/a1; LOG=track_a/logs/a1; mkdir -p "$OUT" "$LOG"
TR=track_a/src/es_train_lora.py
run() { local gpu=$1 sig=$2
  local tag="lora_es_b32_sig${sig}"
  echo "[$(date +%T)] GPU$gpu START sigma=$sig B=32" >> "$LOG/sweep2_driver.log"
  $P4_PY "$TR" --population_size 16 --batch 32 --num_steps 300 --pop_seed 0 \
     --sigma "$sig" --alpha 2e-3 --eval_every 50 --eval_cap 200 --gpu "$gpu" \
     --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE sigma=$sig rc=$?" >> "$LOG/sweep2_driver.log"; }
echo "[$(date +%T)] A1 sweep2 start" > "$LOG/sweep2_driver.log"
( run 0 0.002 ) &
( run 1 0.005 ) &
wait
echo "[$(date +%T)] A1 sweep2 COMPLETE" >> "$LOG/sweep2_driver.log"
