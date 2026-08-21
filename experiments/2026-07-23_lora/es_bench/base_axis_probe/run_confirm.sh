#!/usr/bin/env bash
# Confirmation run: 5 seeds + matched N=12 vanilla control (kills the fewer-explorers confound).
# Reuses pilot seeds {0,1} for the N=16 arms; adds seeds {2,3,4} for them, and N=12 vanilla all 5 seeds.
# Key comparison: base-axis probe (12 Gaussian + 4 anchor) vs N=12 vanilla (12 Gaussian) -> isolates
# the anchor's regularizing contribution at matched Gaussian-explorer count.
set -uo pipefail
cd "$(dirname "$0")/.."
source base_axis_probe/env.sh
OUT=base_axis_probe/results/confirm; LOG=base_axis_probe/logs/confirm
mkdir -p "$OUT" "$LOG"
TRAIN=base_axis_probe/src/es_train_baseaxis.py

run_arm() {  # gpu variant N seed
  local gpu=$1 variant=$2 N=$3 seed=$4
  local tag="${variant}_N${N}_s${seed}"
  echo "[$(date +%T)] GPU$gpu START $tag" >> "$LOG/driver.log"
  $P4_PY "$TRAIN" --variant "$variant" --population_size "$N" --num_steps 200 --pop_seed "$seed" \
     --gpu "$gpu" --a_max 1.0 --anchor_threshold 1.0 --kl --eval_final --eval_cap 300 \
     --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  echo "[$(date +%T)] GPU$gpu DONE  $tag rc=$?" >> "$LOG/driver.log"
}
lane() { local gpu=$1; shift; for s in "$@"; do IFS=: read -r v n sd <<< "$s"; run_arm "$gpu" "$v" "$n" "$sd"; done
         echo "[$(date +%T)] GPU$gpu LANE DONE" >> "$LOG/driver.log"; }

echo "[$(date +%T)] confirm start" > "$LOG/driver.log"
lane 0 baseaxis:16:2 baseaxis:16:3 baseaxis:16:4 vanilla:12:0 vanilla:12:1 &
lane 1 vanilla:16:2 vanilla:16:3 vanilla:16:4 vanilla:12:2 vanilla:12:3 vanilla:12:4 &
wait
echo "[$(date +%T)] confirm ALL COMPLETE" >> "$LOG/driver.log"
