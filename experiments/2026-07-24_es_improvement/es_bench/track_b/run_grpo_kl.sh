#!/usr/bin/env bash
# Item 3 -- KL-to-base for the GRPO MATH checkpoint, on the SAME proxy the ES arms report.
# Two passes (different weight sets can't share one vLLM instance):
#   1) capture base-greedy completions + base logprobs under Qwen2.5-3B-Instruct
#   2) score those exact tokens teacher-forced under the merged GRPO actor
# Prompt set is reconstructed byte-identically to es_train_fullparam.py (MATH L3-5 val[:200]).
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source track_a/env.sh
GPU=${GPU:-0}
BASE=${BASE:-Qwen/Qwen2.5-3B-Instruct}
ACTOR=${ACTOR:-/tmp/grpo_ckpt_grpo_full_3b_math/global_step_200/hf_merged}
OUT=track_b/results/exp3b_math; LOG=track_b/logs/grpo_kl
REC=/tmp/grpo_kl_rec_3b_math.json
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

if [ ! -d "$ACTOR" ]; then log "FATAL: merged actor not found at $ACTOR"; exit 1; fi

log "=== GRPO KL START (base=$BASE actor=$ACTOR) ==="
log "pass 1/2: capture base-greedy completions"
$P4_PY track_b/src/kl_grpo_ckpt.py --mode capture --model "$BASE" --rec "$REC" --gpu $GPU \
  > "$LOG/capture.log" 2>&1 && log "capture DONE" || { log "capture FAIL rc=$?"; exit 1; }

log "pass 2/2: measure drift under the GRPO actor"
$P4_PY track_b/src/kl_grpo_ckpt.py --mode measure --model "$ACTOR" --tokenizer "$BASE" \
  --rec "$REC" --tag grpo_full_3b_math --out "$OUT/grpo_kl_3b_math.json" --gpu $GPU \
  > "$LOG/measure.log" 2>&1 && log "measure DONE" || { log "measure FAIL rc=$?"; exit 1; }

grep -E "^\[capture\]|^\[measure\]" "$LOG"/*.log
log "=== GRPO KL DONE ==="
