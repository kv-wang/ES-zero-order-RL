#!/usr/bin/env bash
# B1: full-param vanilla ES sigma dose-response on GSM8K. 1.5B-Instruct, N=16, 200 steps, 1 seed,
# alpha=5e-4. sigma in {0.5,1,2,4x default(1e-3)}. In-process OOD/ID battery + KL. 2 GPU lanes.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
OUT=track_b/results/b1; LOG=track_b/logs/b1; mkdir -p "$OUT" "$LOG"
TR=track_b/src/es_train_fullparam.py
run() { local gpu=$1 sig=$2
  local tag="fulles_sig${sig}"
  echo "[$(date +%T)] GPU$gpu START sigma=$sig" >> "$LOG/driver.log"
  $P4_PY "$TR" --reward binary --dataset gsm8k --population_size 16 --num_steps 200 \
     --sigma "$sig" --alpha 5e-4 --pop_seed 0 --gpu "$gpu" \
     --eval_final --eval_cap 300 --kl --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE sigma=$sig rc=$?" >> "$LOG/driver.log"; }
echo "[$(date +%T)] B1 sigma sweep start" > "$LOG/driver.log"
( run 0 5e-4; run 0 2e-3 ) &
( run 1 1e-3; run 1 4e-3 ) &
wait
echo "[$(date +%T)] B1 sigma sweep COMPLETE" >> "$LOG/driver.log"
