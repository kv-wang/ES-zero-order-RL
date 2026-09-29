#!/usr/bin/env bash
# A0 LoRA-ES microbench: N in {4,8,16,30} on Qwen-3B, one process per N (clean VRAM), 2 GPU lanes.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
OUT=track_a/results/microbench; LOG=track_a/logs/microbench; mkdir -p "$OUT" "$LOG"
MB=track_a/src/microbench_lora_es.py
run() { local gpu=$1 N=$2
  CUDA_VISIBLE_DEVICES=$gpu $P4_PY "$MB" --pop_size "$N" --out "$OUT/lora_es_N${N}.json" > "$LOG/N${N}.log" 2>&1
  echo "[$(date +%T)] done N=$N rc=$?"; }
( run 0 30; run 0 4 ) &
( run 1 16; run 1 8 ) &
wait
echo "microbench complete"
