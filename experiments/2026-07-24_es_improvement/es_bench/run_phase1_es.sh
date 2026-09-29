#!/usr/bin/env bash
# Phase 1 ES sweep: N in {4,8,16,30}, B=8, 200 steps x 2 seeds.
# Two parallel GPU lanes (seed0 -> GPU0, seed1 -> GPU1), each lane runs the four
# N values sequentially. data_seed + sigma/alpha held constant across all N.
# Config via env: MODEL, DATASET, LEVELS (comma list, math only).
set -uo pipefail
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock
PY=/home/hyin66/micromamba/envs/verl/bin/python
CL=es_bench/commands.log
OUT=es_bench/phase1
LOG=$OUT/logs; mkdir -p "$LOG"

MODEL=${MODEL:?set MODEL}
DATASET=${DATASET:-math}
LEVELS=${LEVELS:-3,4,5}
STEPS=${STEPS:-200}
SIGMA=${SIGMA:-0.001}
ALPHA=${ALPHA:-0.0005}
DATA_SEED=${DATA_SEED:-1234}

run() { # N pop_seed gpu
  local N=$1 S=$2 G=$3
  local tag="N${N}_seed${S}"
  local pfx="$OUT/es_${tag}"
  local cmd="$PY es_bench/es_train.py --model $MODEL --dataset $DATASET --levels $LEVELS \
--population_size $N --batch_size 8 --num_steps $STEPS --sigma $SIGMA --alpha $ALPHA \
--data_seed $DATA_SEED --pop_seed $S --gpu $G --max_tokens 512 --gpu_mem_util 0.5 --out_prefix $pfx"
  echo "[phase1 es] $cmd" >> "$CL"
  echo "[$(date -u +%H:%M:%S)] START $tag on GPU$G"
  $cmd > "$LOG/es_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $tag (exit $?)"
}

lane() { # gpu seed
  local G=$1 S=$2
  for N in 4 8 16 30; do run "$N" "$S" "$G"; done
}

echo "[$(date -u +%H:%M:%S)] PHASE1 ES START model=$MODEL dataset=$DATASET levels=$LEVELS mode=${1:-both}"
if [ "${1:-both}" = "lane" ]; then
  # usage: run_phase1_es.sh lane <gpu> <seed>
  lane "$2" "$3"
else
  lane 0 0 &
  lane 1 1 &
  wait
fi
echo "[$(date -u +%H:%M:%S)] PHASE1 ES DONE (${1:-both})"
