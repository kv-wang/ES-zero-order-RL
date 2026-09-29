#!/usr/bin/env bash
# Phase 1 cheap-decisive diagnostics (single GPU, sequential).
set -uo pipefail
cd "$(dirname "$0")/../.."                 # es_bench
source track_a/env.sh
GPU=${GPU:-0}
D=track_b/diagnostics
OUT=$D/results; LOG=$D/logs; mkdir -p "$OUT" "$LOG"
GRPO_HF=/tmp/grpo_ckpt_grpo_full_3b_math/global_step_200/hf_merged
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

log "=== Phase 1 START ==="

# D1a: base uncapped MATH500 + lengths
log "D1 base start"
$P4_PY $D/d1_run.py --model Qwen/Qwen2.5-3B-Instruct --tag base --gpu $GPU \
  --out $OUT/d1_base.json > "$LOG/d1_base.log" 2>&1 && log "D1 base DONE" || log "D1 base FAIL rc=$?"

# D1b: GRPO uncapped MATH500 + memorization (eval-prompt vs train-prompt on own train set)
log "D1 grpo start"
$P4_PY $D/d1_run.py --model "$GRPO_HF" --tag grpo --gpu $GPU --grpo_train 300 \
  --out $OUT/d1_grpo.json > "$LOG/d1_grpo.log" 2>&1 && log "D1 grpo DONE" || log "D1 grpo FAIL rc=$?"

# D2 full-param reference behavioral scale at sigma=1e-3
log "D2 full start"
$P4_PY $D/d2_probe.py --mode full --gpu $GPU --seeds 8 --sigmas 1e-3 \
  --out $OUT/d2_full.json > "$LOG/d2_full.log" 2>&1 && log "D2 full DONE" || log "D2 full FAIL rc=$?"

# D2 lora sigma grid -> curve + sigma*
log "D2 lora start"
$P4_PY $D/d2_probe.py --mode lora --gpu $GPU --seeds 8 --sigmas 1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1 \
  --out $OUT/d2_lora.json > "$LOG/d2_lora.log" 2>&1 && log "D2 lora DONE" || log "D2 lora FAIL rc=$?"

# D1c: ES-N8 uncapped (deterministic reproduction of the exp3b_math N8 run, eval_cap 500 + lengths)
log "ES-N8 uncapped retrain start"
$P4_PY track_b/src/es_train_fullparam.py --reward binary --model Qwen/Qwen2.5-3B-Instruct --dataset math \
  --population_size 8 --num_steps 200 --sigma 1e-3 --alpha 5e-4 --pop_seed 0 --gpu $GPU \
  --eval_final --eval_cap 500 --kl --out_prefix $OUT/es_N8_uncapped \
  > "$LOG/es_N8_uncapped.log" 2>&1 && log "ES-N8 uncapped DONE" || log "ES-N8 uncapped FAIL rc=$?"

log "=== Phase 1 ALL DONE ==="
