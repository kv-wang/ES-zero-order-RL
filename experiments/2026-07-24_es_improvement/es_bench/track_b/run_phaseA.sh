#!/usr/bin/env bash
# Phase A -- sigma/alpha selection probe (overnight).
# Spec asks for ONE fixed 32-prompt set = 16 GSM8K + 16 MATH L3-5. probe_sigma_scale.py takes a
# single --dataset, so the set is built as two 16-prompt halves and the flip-rates are averaged
# with equal weight -- identical to a single mixed 32-prompt set, and it reuses validated code.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT=track_b/results/overnight/phaseA; LOG=track_b/logs/overnight
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/phaseA.driver.log"; }

log "=== Phase A START (model=$MODEL) ==="
for DS in math gsm8k; do
  log "fullparam reference sigma=1e-3 ($DS, 16 prompts, 8 draws)"
  $P4_PY track_b/src/probe_sigma_scale.py --arm fullparam --model "$MODEL" --dataset $DS \
    --n_prompts 16 --n_seeds 8 --sigmas 1e-3 --gpu $GPU --out "$OUT/ref_${DS}.json" \
    > "$LOG/phaseA_ref_${DS}.log" 2>&1 && log "ref $DS DONE" || log "ref $DS FAIL rc=$?"

  log "lora grid ($DS, 16 prompts, 8 draws, 4 sigmas)"
  $P4_PY track_b/src/probe_sigma_scale.py --arm lora --model "$MODEL" --dataset $DS \
    --n_prompts 16 --n_seeds 8 --sigmas 5e-3,1.5e-2,5e-2,1.5e-1 --gpu $GPU \
    --out "$OUT/lora_${DS}.json" \
    > "$LOG/phaseA_lora_${DS}.log" 2>&1 && log "lora $DS DONE" || log "lora $DS FAIL rc=$?"
done
log "=== Phase A generation DONE -> selecting sigma* ==="
$P4_PY track_b/src/phaseA_select.py --dir "$OUT" --out "$OUT/SIGMA_STAR.json" \
  2>&1 | tee -a "$LOG/phaseA.driver.log"
log "=== Phase A ALL DONE ==="
