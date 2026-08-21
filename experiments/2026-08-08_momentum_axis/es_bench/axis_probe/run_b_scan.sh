#!/usr/bin/env bash
set -e
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-08-08_momentum_axis/es_bench
source axis_probe/env.sh

OUT=axis_probe/results/b_scan_base
LOG=axis_probe/logs/b_scan_base
TRAIN=axis_probe/src/es_train_axis.py
mkdir -p "$OUT" "$LOG"

for B in 16 32 64; do
    TAG="vanilla_B${B}_N16_s0"
    echo "[$(date +%F_%T)] START B=$B"
    $P4_PY "$TRAIN" \
        --variant vanilla \
        --population_size 16 \
        --batch "$B" \
        --num_steps 50 \
        --pop_seed 0 \
        --gpu 0 \
        --kl \
        --eval_final \
        --eval_cap 300 \
        --out_prefix "$OUT/$TAG" \
        > "$LOG/${TAG}.log" 2>&1
    echo "[$(date +%F_%T)] DONE  B=$B rc=$?"
done
echo "[$(date +%F_%T)] ALL B SCAN DONE"
