#!/usr/bin/env bash
# Launch full 100-step countdown training on Qwen2.5-3B-Instruct
# This script wraps run_countdown_3b.sh with production settings.
#
# Usage:
#   ./launch_countdown_3b_full.sh              # Single seed=0, all stages
#   SEEDS="0 1 2" ./launch_countdown_3b_full.sh  # Multi-seed
#   STAGES="2 3" ./launch_countdown_3b_full.sh   # Skip base/readout, only ES arms

set -euo pipefail
cd "$(dirname "$0")/.."

# Production defaults (override via environment)
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export STAGES="${STAGES:-0 2 3 4}"
export SEEDS="${SEEDS:-0}"
export GPU="${GPU:-0}"
export TP="${TP:-1}"

# ES hyperparameters (match 7B paper configuration)
export N="${N:-30}"
export B="${B:-100}"
export STEPS="${STEPS:-100}"
export EVAL_EVERY="${EVAL_EVERY:-10}"
export SIGMA="${SIGMA:-1e-3}"
export ALPHA="${ALPHA:-5e-4}"
export MINI_BATCH="${MINI_BATCH:-64}"

# Evaluation settings
export EVAL_CAP="${EVAL_CAP:-300}"
export MAXTOK="${MAXTOK:-2048}"

# Memory and diagnostics
export ES_MEM_UTIL="${ES_MEM_UTIL:-0.70}"
export MEM_POLL="${MEM_POLL:-10}"
export RHO_SPLITS="${RHO_SPLITS:-20}"

# Timing calibration (update after smoke test completes)
export S_PER_GEN="${S_PER_GEN:-0.15}"

# Output directory
TAG="countdown_3b_N${N}_B${B}"
export CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"

echo "================================================================================"
echo "  Launching FULL 3B countdown ES training"
echo "================================================================================"
echo "Model     : $MODEL"
echo "GPU       : $GPU (single card, TP=$TP)"
echo "Stages    : $STAGES"
echo "Seeds     : $SEEDS"
echo "ES config : N=$N B=$B steps=$STEPS sigma=$SIGMA alpha=$ALPHA"
echo "Eval      : every $EVAL_EVERY steps, cap=$EVAL_CAP questions"
echo "Results   : axis_probe/results/${TAG}"
echo "Checkpoints: $CKPT"
echo "================================================================================"
echo ""

read -p "Proceed with full training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

# Execute the main driver
exec ./axis_probe/run_countdown_3b.sh
