#!/usr/bin/env bash
# Phase 3: minimum viable set with seeds + KL proxy + in-process eval.
#   small N {2,4}: seeds {0,1,2} (seed0 re-run adds KL + cross-checks Phase 2)
#   N=30 control : seeds {1,2}   (seed0 accuracy already in Phase 2; KL at 2 seeds)
# 16 ES arms, 2 load-balanced lanes (one per H200). Small-N first (priority data
# lands early), N=30 long poles last. GRPO handled separately.
set -uo pipefail
cd "$(dirname "$0")/.."
source phase4_continuous_ood/env.sh
OUT=phase4_continuous_ood/results/phase3; LOG=phase4_continuous_ood/logs/phase3
mkdir -p "$OUT" "$LOG"
TRAIN=phase4_continuous_ood/src/es_train_continuous.py

run_arm() {  # gpu reward N seed
  local gpu=$1 reward=$2 N=$3 seed=$4
  local tag="${reward}_N${N}_s${seed}"
  echo "[$(date +%T)] GPU$gpu START $tag" >> "$LOG/driver.log"
  $P4_PY "$TRAIN" --reward "$reward" --population_size "$N" --num_steps 200 \
     --pop_seed "$seed" --gpu "$gpu" --kl --eval_final --eval_cap 300 \
     --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE  $tag rc=$?" >> "$LOG/driver.log"
}

lane() {  # gpu "reward:N:seed ..."
  local gpu=$1; shift
  for s in "$@"; do IFS=: read -r r n sd <<< "$s"; run_arm "$gpu" "$r" "$n" "$sd"; done
  echo "[$(date +%T)] GPU$gpu LANE COMPLETE" >> "$LOG/driver.log"
}

echo "[$(date +%T)] Phase 3 start" > "$LOG/driver.log"
lane 0 continuous:4:0 continuous:4:1 continuous:4:2 continuous:2:0 continuous:2:1 continuous:2:2 continuous:30:1 binary:30:2 &
lane 1 binary:4:0 binary:4:1 binary:4:2 binary:2:0 binary:2:1 binary:2:2 binary:30:1 continuous:30:2 &
wait
echo "[$(date +%T)] Phase 3 ALL COMPLETE" >> "$LOG/driver.log"
