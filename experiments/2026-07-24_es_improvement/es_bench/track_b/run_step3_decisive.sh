#!/usr/bin/env bash
# Step 3 -- decisive LoRA-ES re-tune run (1.5B, MATH L3-5, 300 steps, N=16, B=8, seed 0).
#
# Three arms. Single-variable discipline: N, B, shaping, greedy fitness, data seed all fixed;
# only (sigma, alpha) move.
#   control : sigma=5e-3   alpha=2e-3   -- the BEFORE-FIX setting, run at matched N and step
#                                          count so the gate threshold is measured here rather
#                                          than borrowed from the 3B N=16 arm (+0.022).
#   A       : sigma*=7.7e-3 alpha=1e-2  -- alpha* by the Step-2 KL-velocity gate (7.1x N30).
#   B       : sigma*=7.7e-3 alpha=3e-3  -- alpha targeting the middle of the Step-3 KL band.
#
# A and B bracket a conflict between the two gates: the Step-2 velocity criterion averages KL
# over 20 steps while the Step-3 band applies at 300, and KL is superlinear in t (p=1.56 at
# alpha=1e-2), so A is predicted to overshoot the band and B to land inside it. Running both
# settles it empirically in one pass.
#
# GATE: train fitness rises clearly beyond the control AND KL in [1e-3, 3e-2].
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
STEPS=${STEPS:-300}
N=${N:-16}
OUT=track_b/results/step3_decisive; LOG=track_b/logs/step3_decisive
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

run_arm(){  # tag sigma alpha
  local tag=$1 sig=$2 alp=$3
  log "$tag start (sigma=$sig alpha=$alp)"
  $P4_PY track_a/src/es_train_lora.py --model "$MODEL" --dataset math \
    --population_size $N --batch 8 --num_steps $STEPS \
    --sigma "$sig" --alpha "$alp" --pop_seed 0 --gpu $GPU \
    --eval_every 50 --eval_cap 200 --no_final_eval \
    --out_prefix "$OUT/$tag" > "$LOG/${tag}.log" 2>&1 \
    && log "$tag DONE" || log "$tag FAIL rc=$?"
}

log "=== Step3 decisive START (model=$MODEL N=$N steps=$STEPS) ==="
run_arm control_s5e-3_a2e-3 0.005  2e-3
run_arm B_sigstar_a3e-3     0.0077 3e-3
run_arm A_sigstar_a1e-2     0.0077 1e-2
log "=== Step3 decisive ALL DONE ==="
