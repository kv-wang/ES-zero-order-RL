#!/usr/bin/env bash
# Run Qwen2.5-3B (base, not Instruct) on MATH task with three methods:
#   stage 0   untrained base reference
#   stage 1   GRPO (400 steps, generation-matched)
#   stage 2   ES vanilla (N=16, 200 steps, seed 0)
#   stage 3   ES baseaxis (N=16, 200 steps, seed 0)
#
# All checkpoints are saved to /tmp/qwen3b_base_ckpts/ on local disk.
#
# Usage:
#   ./run_qwen3b_base.sh                    # full run, ~8-10h
#   SMOKE=1 ./run_qwen3b_base.sh            # smoke test (10 ES steps, 20 GRPO steps)
#   STAGES="0 1" ./run_qwen3b_base.sh       # just base + GRPO
set -euo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

# ---- configuration ----
MODEL="Qwen/Qwen2.5-3B"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"
STAGES="${STAGES:-0 1 2 3}"
SEEDS="${SEEDS:-0}"                           # single seed for initial run

# Checkpoint directory on local disk (not home quota)
CKPT_BASE="/tmp/qwen3b_base_ckpts"
mkdir -p "$CKPT_BASE"

if [[ "$SMOKE" == "1" ]]; then
  EVAL_CAP=40
  export STEPS=10 N=8
  GRPO_STEPS=20
else
  EVAL_CAP=300
  export STEPS=200 N=16
  GRPO_STEPS=400
fi

# Results directory
RESULTS_DIR="axis_probe/results/qwen3b_base"
LOG_DIR="axis_probe/logs/qwen3b_base"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/driver.log"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER_LOG"; }

say "=== Qwen2.5-3B base training start ==="
say "model=$MODEL  gpu=$GPU  smoke=$SMOKE  stages=[$STAGES]  seeds=[$SEEDS]"
say "eval_cap=$EVAL_CAP  ES: N=$N steps=$STEPS  GRPO: steps=$GRPO_STEPS"
say "checkpoints -> $CKPT_BASE"
say "GPU $GPU free memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$GPU" 2>/dev/null || echo 'N/A')"

stage_wanted() { [[ " $STAGES " == *" $1 "* ]]; }

run_stage() {
  local id=$1 label=$2; shift 2
  stage_wanted "$id" || { say "SKIP  stage $id ($label) -- not in STAGES"; return 0; }
  say "START stage $id: $label"
  local t0=$SECONDS rc=0
  "$@" || rc=$?
  say "DONE  stage $id: $label rc=$rc elapsed=$((SECONDS-t0))s"
  [[ $rc -ne 0 ]] && say "  !! stage $id FAILED -- stopping" && exit $rc
  return 0
}

# ---- stage 0: untrained base reference ----
stage_base() {
  local out="$RESULTS_DIR/base_summary.json"
  if [[ -f "$out" ]]; then
    say "  base summary exists, skipping"
    return 0
  fi
  say "  evaluating base model..."
  MODEL="$MODEL" $P4_PY axis_probe/eval_base.py \
    --out_prefix "$RESULTS_DIR/base" \
    --eval_cap "$EVAL_CAP" \
    --gpu "$GPU" \
    > "$LOG_DIR/base.log" 2>&1
}

# ---- stage 1: GRPO ----
stage_grpo() {
  local ckpt_dir="$CKPT_BASE/grpo_step${GRPO_STEPS}"
  local out="$RESULTS_DIR/grpo_math_gen${GRPO_STEPS}_summary.json"

  if [[ -f "$out" ]]; then
    say "  GRPO summary exists, skipping"
    return 0
  fi

  say "  training GRPO (${GRPO_STEPS} steps)..."
  MODEL="$MODEL" \
  GPU="$GPU" \
  EVAL_CAP="$EVAL_CAP" \
  BUDGET=manual \
  STEPS="$GRPO_STEPS" \
  CKPT="$ckpt_dir" \
    ./axis_probe/run_grpo.sh > "$LOG_DIR/grpo.log" 2>&1

  say "  GRPO checkpoint saved to: $ckpt_dir"
}

# ---- stage 2: ES vanilla ----
stage_es_vanilla() {
  for seed in $SEEDS; do
    local ckpt_dir="$CKPT_BASE/vanilla_N${N}_s${seed}"
    local out="$RESULTS_DIR/vanilla_N${N}_s${seed}_summary.json"

    if [[ -f "$out" ]]; then
      say "  ES vanilla seed=$seed summary exists, skipping"
      continue
    fi

    say "  training ES vanilla (N=$N, steps=$STEPS, seed=$seed)..."
    MODEL="$MODEL" \
    GPU="$GPU" \
    EVAL_CAP="$EVAL_CAP" \
    VARIANTS="vanilla" \
    SEEDS="$seed" \
    SAVE_CHECKPOINTS="$ckpt_dir" \
      ./axis_probe/run_momentum.sh > "$LOG_DIR/vanilla_s${seed}.log" 2>&1

    say "  ES vanilla seed=$seed checkpoint saved to: $ckpt_dir"
  done
}

# ---- stage 3: ES baseaxis ----
stage_es_baseaxis() {
  for seed in $SEEDS; do
    local ckpt_dir="$CKPT_BASE/baseaxis_N${N}_s${seed}"
    local out="$RESULTS_DIR/baseaxis_N${N}_s${seed}_summary.json"

    if [[ -f "$out" ]]; then
      say "  ES baseaxis seed=$seed summary exists, skipping"
      continue
    fi

    say "  training ES baseaxis (N=$N, steps=$STEPS, seed=$seed)..."
    MODEL="$MODEL" \
    GPU="$GPU" \
    EVAL_CAP="$EVAL_CAP" \
    VARIANTS="baseaxis" \
    SEEDS="$seed" \
    SAVE_CHECKPOINTS="$ckpt_dir" \
      ./axis_probe/run_momentum.sh > "$LOG_DIR/baseaxis_s${seed}.log" 2>&1

    say "  ES baseaxis seed=$seed checkpoint saved to: $ckpt_dir"
  done
}

# ---- execute stages ----
run_stage 0 "untrained base reference" stage_base
run_stage 1 "GRPO" stage_grpo
run_stage 2 "ES vanilla" stage_es_vanilla
run_stage 3 "ES baseaxis" stage_es_baseaxis

say "=== All stages complete ==="
say "Results directory: $RESULTS_DIR"
say "Checkpoints directory: $CKPT_BASE"
say ""
say "Summary files:"
ls -lh "$RESULTS_DIR"/*_summary.json 2>/dev/null || say "  (none found yet)"
say ""
say "Total size of checkpoints:"
du -sh "$CKPT_BASE" 2>/dev/null || say "  (directory not found)"
