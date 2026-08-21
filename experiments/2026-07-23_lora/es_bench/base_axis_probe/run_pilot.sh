#!/usr/bin/env bash
# Base-axis probe pilot: probe vs matched vanilla control. N=16, seeds {0,1}, 200 steps,
# a_max=1.0, binary reward, KL + OOD/ID eval. Two lanes, one per H200.
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source base_axis_probe/env.sh
OUT=base_axis_probe/results/pilot; LOG=base_axis_probe/logs/pilot
mkdir -p "$OUT" "$LOG"
TRAIN=base_axis_probe/src/es_train_baseaxis.py

run_arm() {  # gpu variant seed
  local gpu=$1 variant=$2 seed=$3
  local tag="${variant}_N16_s${seed}"
  echo "[$(date +%T)] GPU$gpu START $tag" >> "$LOG/driver.log"
  $P4_PY "$TRAIN" --variant "$variant" --population_size 16 --num_steps 200 --pop_seed "$seed" \
     --gpu "$gpu" --a_max 1.0 --anchor_threshold 1.0 --kl --eval_final --eval_cap 300 \
     --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE  $tag rc=$?" >> "$LOG/driver.log"
}

echo "[$(date +%T)] base-axis pilot start" > "$LOG/driver.log"
( run_arm 0 baseaxis 0; run_arm 0 baseaxis 1; echo "[$(date +%T)] GPU0 LANE DONE" >> "$LOG/driver.log" ) &
( run_arm 1 vanilla  0; run_arm 1 vanilla  1; echo "[$(date +%T)] GPU1 LANE DONE" >> "$LOG/driver.log" ) &
wait
echo "[$(date +%T)] base-axis pilot ALL COMPLETE" >> "$LOG/driver.log"
