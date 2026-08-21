#!/usr/bin/env bash
# Phase 3: dual-GPU ES scaling. Runs es_train_dual (population split 2+2 / 4+4)
# on GPUs 0+1 for N in {4,8}, same 3B model + settings as Phase 2 so the
# single-GPU baseline can be reused from Phase 2 (es_N4/es_N8 summaries).
set -uo pipefail
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock
PY=/home/hyin66/micromamba/envs/verl/bin/python
CL=es_bench/commands.log
OUT=es_bench/phase3; LOG=$OUT/logs; mkdir -p "$LOG"
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
MAXTOK=${MAXTOK:-512}
STEPS=${STEPS:-30}
export XDG_CONFIG_HOME=/tmp/xdgconfig VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$XDG_CONFIG_HOME"

for N in 4 8; do
  cmd="$PY es_bench/es_train_dual.py --model $MODEL --dataset math --levels 3,4,5 \
--population_size $N --batch_size 8 --num_steps $STEPS --warmup 5 --sigma 0.001 --alpha 0.0005 \
--data_seed 1234 --pop_seed 0 --gpus 0,1 --max_tokens $MAXTOK --gpu_mem_util 0.5 --out_prefix $OUT/dual_N${N}"
  echo "[phase3 dual N=$N] $cmd" >> "$CL"
  echo "[$(date -u +%H:%M:%S)] START phase3 dual N=$N"
  $cmd > "$LOG/dual_N${N}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE phase3 dual N=$N (exit $?)"
done
echo "[$(date -u +%H:%M:%S)] PHASE3 DUAL DONE"
