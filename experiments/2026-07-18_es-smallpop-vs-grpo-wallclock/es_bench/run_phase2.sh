#!/usr/bin/env bash
# Phase 2: single-step wall-clock, 3 arms, SAME GPU0 + 3B model + shared reward +
# same max_new_tokens, all vLLM. Timing runs use >=25 steps (5 warmup discarded).
#   a) ES N=4, B=8  (32 seqs/step)
#   b) ES N=8, B=8  (64 seqs/step, generation volume aligned with GRPO)
#   c) GRPO default (report actual seqs/step)
set -uo pipefail
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock
PY=/home/hyin66/micromamba/envs/verl/bin/python
CL=es_bench/commands.log
OUT=es_bench/phase2; LOG=$OUT/logs; mkdir -p "$LOG"
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
MAXTOK=${MAXTOK:-512}
STEPS=${STEPS:-30}

es_arm() { # N tag
  local N=$1 tag=$2
  local cmd="$PY es_bench/es_train.py --model $MODEL --dataset math --levels 3,4,5 \
--population_size $N --batch_size 8 --num_steps $STEPS --warmup 5 --sigma 0.001 --alpha 0.0005 \
--data_seed 1234 --pop_seed 0 --gpu 0 --max_tokens $MAXTOK --gpu_mem_util 0.5 --out_prefix $OUT/es_${tag}"
  echo "[phase2 $tag] $cmd" >> "$CL"
  echo "[$(date -u +%H:%M:%S)] START phase2 $tag"; $cmd > "$LOG/es_${tag}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE phase2 $tag (exit $?)"
}

echo "[$(date -u +%H:%M:%S)] PHASE2 START model=$MODEL maxtok=$MAXTOK steps=$STEPS"
es_arm 4 N4
es_arm 8 N8
# GRPO arm: 3B, group=8, train_batch=8 -> 64 rollouts/step, same GPU0, max_response=MAXTOK
echo "[phase2 grpo] run_grpo 3B GPU0 STEPS=$STEPS group=8" >> "$CL"
# sample GPU0 peak memory during the GRPO run
( peak=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1);
  [ -n "$u" ] && [ "$u" -gt "$peak" ] && { peak=$u; echo "$peak" > "$OUT/grpo_gpu0_peak_mib.txt"; }; sleep 0.5; done ) &
SAMPLER=$!
MODEL=$MODEL GPU=0 STEPS=$STEPS GROUP=8 TRAIN_BS=8 MINI_BS=8 MICRO=8 \
  EXP=phase2_grpo MAXRESP=$MAXTOK bash es_bench/grpo/run_grpo.sh > "$LOG/grpo.log" 2>&1
kill $SAMPLER 2>/dev/null
echo "[$(date -u +%H:%M:%S)] PHASE2 GRPO done (exit $?) peak_gpu0=$(cat $OUT/grpo_gpu0_peak_mib.txt 2>/dev/null)MiB"
echo "[$(date -u +%H:%M:%S)] PHASE2 ALL DONE"
