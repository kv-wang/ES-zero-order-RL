#!/usr/bin/env bash
# Step 2 -- 20-step KL-velocity probe for LoRA-ES.
#
# Single-variable discipline: sigma is PINNED at sigma* from D2 (geometric mean of the
# flip-rate and KL matches to the full-param sigma=1e-3 reference); N=16, B=8, vanilla
# z-score shaping, greedy fitness -- only alpha varies.
#
# Target band: per-step KL velocity in 1-10x full-param ES N=30, whose 200-step run gave
# KL 29.1e-3 -> 1.455e-4 per step. So alpha* should land KL_velocity in [1.46e-4, 1.46e-3].
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
SIGMA=${SIGMA:-0.0077}                        # sigma* from results/d2_probe/D2_SIGMA_STAR.json
STEPS=${STEPS:-20}
N=${N:-16}
OUT=track_b/results/step2_alpha; LOG=track_b/logs/step2_alpha
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

log "=== Step2 alpha probe START (model=$MODEL sigma*=$SIGMA N=$N steps=$STEPS) ==="
for A in 2e-3 1e-2 5e-2 2e-1; do
  tag="a${A}"
  log "alpha=$A start"
  $P4_PY track_a/src/es_train_lora.py --model "$MODEL" --dataset math \
    --population_size $N --batch 8 --num_steps $STEPS \
    --sigma "$SIGMA" --alpha "$A" --pop_seed 0 --gpu $GPU \
    --eval_every 10 --eval_cap 200 --kl_only --no_final_eval \
    --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1 \
    && log "alpha=$A DONE" || log "alpha=$A FAIL rc=$?"
done
log "=== Step2 alpha probe ALL DONE ==="
