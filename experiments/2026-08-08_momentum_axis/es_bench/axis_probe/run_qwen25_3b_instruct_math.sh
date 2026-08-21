#!/usr/bin/env bash
# Qwen2.5-3B-Instruct math post-training comparison.
#
# Runs the requested methods on the existing MATH L3-5 axis_probe harness:
#   stage 0  untrained base reference eval
#   stage 1  GRPO
#   stage 2  vanilla ES
#   stage 3  base-axis ES
#   stage 5  combined readout
#
# Thin wrapper around run_all_arms.sh so data split, prompt builder, reward, eval battery,
# resume behavior, and model-scoped result directories remain single-sourced.
#
# Usage:
#   ./axis_probe/run_qwen25_3b_instruct_math.sh
#   SEEDS="0 1" ./axis_probe/run_qwen25_3b_instruct_math.sh
#   SMOKE=1 ./axis_probe/run_qwen25_3b_instruct_math.sh
set -uo pipefail

cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export TASK="${TASK:-math}"
export GPU="${GPU:-0}"
export SEEDS="${SEEDS:-0}"
export N="${N:-16}"
export STEPS="${STEPS:-200}"
export EVAL_CAP="${EVAL_CAP:-300}"
export SMOKE="${SMOKE:-0}"

# Full-param ES keeps an fp32 base copy, and base-axis keeps an additional fp32 theta0 copy.
# 3B should fit comfortably on one H200; this leaves extra headroom without changing ES math.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"

# Keep GRPO close to the existing comparable defaults; override if a memory issue appears.
export ROLLOUT_MEM="${ROLLOUT_MEM:-0.50}"
export MICRO="${MICRO:-4}"

# run_grpo.sh matches GRPO to one ES arm by total rollout count.
export ES_BATCH="${ES_BATCH:-$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.BATCH_SIZE)")}"
export ES_GENERATIONS="${ES_GENERATIONS:-$((STEPS * N * ES_BATCH))}"

# Requested methods only: base reference, GRPO, vanilla ES, base-axis ES, final readout.
export STAGES="${STAGES:-0 1 2 3 5}"

echo "[$(date +%F_%T)] qwen25-3b-instruct math run"
echo "  MODEL=$MODEL"
echo "  TASK=$TASK STAGES=[$STAGES] SEEDS=[$SEEDS] N=$N STEPS=$STEPS ES_BATCH=$ES_BATCH"
echo "  GPU=$GPU EVAL_CAP=$EVAL_CAP SMOKE=$SMOKE ES_GENERATIONS=$ES_GENERATIONS"
echo "  GPU_MEM_UTIL=$GPU_MEM_UTIL ROLLOUT_MEM=$ROLLOUT_MEM MICRO=$MICRO"

exec ./axis_probe/run_all_arms.sh
