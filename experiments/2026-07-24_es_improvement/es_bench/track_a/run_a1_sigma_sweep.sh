#!/usr/bin/env bash
# A1: LoRA-ES sigma 3-point sweep (exploration radius). 1.5B-Instruct, GSM8K, N=16, 300 steps,
# 1 seed, alpha=2e-3 fixed. 2 GPU lanes (LoRA-ES parallelizes; no ray).
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
OUT=track_a/results/a1; LOG=track_a/logs/a1; mkdir -p "$OUT" "$LOG"
TR=track_a/src/es_train_lora.py
run() { local gpu=$1 sig=$2
  local tag="lora_es_sig${sig}"
  echo "[$(date +%T)] GPU$gpu START sigma=$sig" >> "$LOG/sweep_driver.log"
  CUDA_VISIBLE_DEVICES=$gpu $P4_PY "$TR" --population_size 16 --num_steps 300 --pop_seed 0 \
     --sigma "$sig" --alpha 2e-3 --eval_every 50 --eval_cap 200 --gpu "$gpu" \
     --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE sigma=$sig rc=$?" >> "$LOG/sweep_driver.log"; }
echo "[$(date +%T)] A1 sigma sweep start" > "$LOG/sweep_driver.log"
( run 0 0.01; run 0 0.05 ) &
( run 1 0.02 ) &
wait
echo "[$(date +%T)] A1 sigma sweep COMPLETE" >> "$LOG/sweep_driver.log"
