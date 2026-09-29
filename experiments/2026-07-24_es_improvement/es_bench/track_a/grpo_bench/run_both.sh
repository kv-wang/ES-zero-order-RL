#!/usr/bin/env bash
# Run the two GRPO benches SEQUENTIALLY (parallel verl instances OOM the node's CPU RAM).
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=0 LORA_RANK=0  EXP=grpo_full STEPS=25 bash track_a/grpo_bench/run_grpo_bench.sh
GPU=0 LORA_RANK=16 EXP=grpo_lora STEPS=25 bash track_a/grpo_bench/run_grpo_bench.sh
echo "BOTH DONE"
