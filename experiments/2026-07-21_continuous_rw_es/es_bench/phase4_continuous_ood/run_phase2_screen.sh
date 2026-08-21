#!/usr/bin/env bash
# Phase 2 seed-0 screen: 10 ES arms (binary+continuous x N in {2,4,6,8,30}),
# full 200-step budget, in-process final OOD/ID eval. Two load-balanced lanes,
# one per H200 (sum of N per lane = 50). GRPO handled separately after. seed 0.
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source phase4_continuous_ood/env.sh
OUT=phase4_continuous_ood/results/phase2; LOG=phase4_continuous_ood/logs/phase2
mkdir -p "$OUT" "$LOG"
TRAIN=phase4_continuous_ood/src/es_train_continuous.py
SEED=0

run_arm() {  # gpu reward N
  local gpu=$1 reward=$2 N=$3
  local tag="${reward}_N${N}_s${SEED}"
  echo "[$(date +%T)] GPU$gpu START $tag" >> "$LOG/driver.log"
  # trainer masks to physical GPU via --gpu (it sets CUDA_VISIBLE_DEVICES itself)
  $P4_PY "$TRAIN" \
     --reward "$reward" --population_size "$N" --num_steps 200 --pop_seed $SEED --gpu "$gpu" \
     --eval_final --eval_cap 300 \
     --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE  $tag rc=$?" >> "$LOG/driver.log"
}

lane() {  # gpu  "reward:N reward:N ..."
  local gpu=$1; shift
  for spec in "$@"; do run_arm "$gpu" "${spec%%:*}" "${spec##*:}"; done
  echo "[$(date +%T)] GPU$gpu LANE COMPLETE" >> "$LOG/driver.log"
}

echo "[$(date +%T)] Phase 2 screen start" > "$LOG/driver.log"
# N=30 first (long pole) so the critical path starts immediately.
lane 0  continuous:30 continuous:8 continuous:4 continuous:2 binary:6  &
lane 1  binary:30     binary:8     binary:4     binary:2     continuous:6 &
wait
echo "[$(date +%T)] Phase 2 screen ALL COMPLETE" >> "$LOG/driver.log"
