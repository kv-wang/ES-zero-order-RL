#!/usr/bin/env bash
# Full mirror of the GSM8K 3B comparison, but trained on MATH L3-5 (proper difficulty),
# with Countdown added to the OOD battery. Single GPU, sequential.
#   arms: base | full-param ES N{4,8,20,30} | LoRA-ES N16 | GRPO   (all Qwen2.5-3B-Instruct)
set -uo pipefail
cd "$(dirname "$0")/.."                      # es_bench
source track_a/env.sh
GPU=${GPU:-0}
MODEL=Qwen/Qwen2.5-3B-Instruct
OUT=track_b/results/exp3b_math
LOG=track_b/logs/exp3b_math
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

log "=== exp3b_math START (model=$MODEL) ==="

# 1) base anchor (battery + countdown)
if [ ! -f "$OUT/base_3b.json" ]; then
  log "BASE eval start"
  CUDA_VISIBLE_DEVICES=$GPU $P4_PY track_b/src/eval_base.py --model "$MODEL" --eval_cap 300 \
    --out "$OUT/base_3b.json" > "$LOG/base_eval.log" 2>&1 && log "BASE eval DONE" || log "BASE eval FAIL rc=$?"
fi

# 2) LoRA-ES N16 (fast)
log "LoRA-ES N16 start"
MODEL="$MODEL" DATASET=math OUTDIR="$OUT" TAGPRE=loraes STEPS=200 \
  bash track_b/run_exp_es.sh $GPU loraes 16 && log "LoRA-ES DONE" || log "LoRA-ES FAIL rc=$?"

# 3) GRPO full-param (train -> FSDP merge -> eval)
EXP=grpo_full_3b_math; CKPT=/tmp/grpo_ckpt_$EXP; rm -rf "$CKPT"
log "GRPO train start ($EXP)"
MODEL="$MODEL" GPU=$GPU STEPS=200 EXP=$EXP \
  DATA_DIR="$(pwd)/track_b/grpo_ood/data_math" \
  bash track_b/grpo_ood/run_grpo_train.sh > "$LOG/grpo_train.log" 2>&1 && log "GRPO train DONE" || log "GRPO train FAIL rc=$?"
log "GRPO eval start"
$P4_PY track_b/grpo_ood/eval_grpo_ood.py --actor_dir "$CKPT/global_step_200/actor" \
  --tag grpo_full_3b_math --out "$OUT/grpo_full_3b.json" --eval_cap 300 --gpu $GPU \
  > "$LOG/grpo_eval.log" 2>&1 && log "GRPO eval DONE" || log "GRPO eval FAIL rc=$?"

# 4) full-param ES sweep (N=4,8,20,30) — slowest last
for N in 4 8 20 30; do
  log "full-param ES N$N start"
  MODEL="$MODEL" DATASET=math OUTDIR="$OUT" TAGPRE=fulles STEPS=200 \
    bash track_b/run_exp_es.sh $GPU fulles $N && log "full-param ES N$N DONE" || log "full-param ES N$N FAIL rc=$?"
done

log "=== exp3b_math ALL DONE ==="
