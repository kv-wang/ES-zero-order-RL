#!/usr/bin/env bash
# Phase 0: base GSM8K accuracy (500 Q, greedy pass@1) for model selection.
# Runs the two 1.5B variants in parallel (GPU0/GPU1), then the 3B on GPU0.
# NOTE: Qwen2.5-Math-3B does NOT exist upstream, so 3B has only the Instruct variant.
set -uo pipefail
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock
PY=/home/hyin66/micromamba/envs/verl/bin/python
export HF_HUB_DISABLE_XET=1 HF_DATASETS_CACHE=/home/hyin66/.cache/hf_datasets
R=es_bench/phase0/results
L=es_bench/phase0/logs
CL=es_bench/commands.log
N=500

run() { # model gpu tag
  echo "[phase0 eval] $PY es_bench/phase0/gsm8k_eval.py --model $1 --n $N --gpu $2 --out_prefix $R/gsm8k_$3" >> "$CL"
  $PY es_bench/phase0/gsm8k_eval.py --model "$1" --n "$N" --gpu "$2" --out_prefix "$R/gsm8k_$3" > "$L/gsm8k_$3.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] done $3 (exit $?)"
}

echo "[$(date -u +%H:%M:%S)] START phase0 GSM8K evals"
run Qwen/Qwen2.5-Math-1.5B-Instruct 0 math1.5b &
run Qwen/Qwen2.5-1.5B-Instruct      1 instruct1.5b &
wait
run Qwen/Qwen2.5-3B-Instruct        0 instruct3b
echo "[$(date -u +%H:%M:%S)] ALL phase0 evals done"
echo "=== SUMMARIES ==="
for f in $R/gsm8k_*_summary.json; do echo "--- $f ---"; cat "$f"; echo; done
