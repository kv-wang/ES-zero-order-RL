#!/usr/bin/env bash
# Qwen2.5-7B-Instruct math post-training comparison.
#
# Runs the three requested training methods on the existing MATH L3-5 axis_probe harness:
#   stage 0  untrained base reference eval
#   stage 1  GRPO
#   stage 2  vanilla ES
#   stage 3  base-axis ES
#   stage 5  combined readout
#
# This is deliberately a thin wrapper around run_all_arms.sh. The delegated scripts already
# define the shared data split, prompt builder, binary math reward, eval battery, checkpoint
# resume behavior, and model-scoped result suffixes. Keeping those definitions single-sourced
# avoids a 7B-specific fork drifting from the main comparison protocol.
#
# Defaults are a full single-seed run. Override from the environment for repeats/smokes:
#   SEEDS="0 1" ./axis_probe/run_qwen25_7b_instruct_math.sh
#   SMOKE=1 ./axis_probe/run_qwen25_7b_instruct_math.sh
#   GPU=1 EVAL_CAP=100 ./axis_probe/run_qwen25_7b_instruct_math.sh
set -uo pipefail

cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TASK="${TASK:-math}"
export GPU="${GPU:-0}"
export SEEDS="${SEEDS:-0}"
export N="${N:-16}"
export STEPS="${STEPS:-200}"
export EVAL_CAP="${EVAL_CAP:-300}"
export SMOKE="${SMOKE:-0}"

# 7B full-parameter ES stores one fp32 base copy for vanilla and two fp32 copies for base-axis.
# Keep vLLM KV/cudagraph reservation lower than the 1.5B default so those buffers fit on one H200.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"

# GRPO at the 1.5B defaults OOMs on 7B during the Adam step when rollout memory is 0.5.
# Lower the colocated vLLM reservation while keeping the generation-matched step count intact.
export ROLLOUT_MEM="${ROLLOUT_MEM:-0.25}"
export MICRO="${MICRO:-1}"

# run_grpo.sh matches GRPO to an ES arm by total rollout count. If N/STEPS are overridden,
# keep the GRPO generation budget matched to the ES arm shape used below.
export ES_BATCH="${ES_BATCH:-$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.BATCH_SIZE)")}"
export ES_GENERATIONS="${ES_GENERATIONS:-$((STEPS * N * ES_BATCH))}"

# Requested methods only: base reference, GRPO, vanilla ES, base-axis ES, final readout.
export STAGES="${STAGES:-0 1 2 3 5}"

echo "[$(date +%F_%T)] qwen25-7b-instruct math run"
echo "  MODEL=$MODEL"
echo "  TASK=$TASK STAGES=[$STAGES] SEEDS=[$SEEDS] N=$N STEPS=$STEPS ES_BATCH=$ES_BATCH"
echo "  GPU=$GPU EVAL_CAP=$EVAL_CAP SMOKE=$SMOKE ES_GENERATIONS=$ES_GENERATIONS"
echo "  GPU_MEM_UTIL=$GPU_MEM_UTIL ROLLOUT_MEM=$ROLLOUT_MEM MICRO=$MICRO"

exec ./axis_probe/run_all_arms.sh
