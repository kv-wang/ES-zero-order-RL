#!/usr/bin/env bash
# Phase 1 -- tail averaging vs pure shrinkage, on the 3B MATH L3-5 full-param ES arms.
#
# No retraining: each arm's trajectory is reconstructed exactly from its logged fitness
# vectors + the pop_seed stream (tail_average.py / tailavg_worker.py). Every run is
# gated on reproducing the logged per-step ||Delta_s|| before any eval is reported.
#
# Per arm the variants are:
#   theta_final                 the endpoint (lambda=1) -- also the replay cross-check,
#                               since its eval must match the value recorded at training time
#   tailavg_avg5                mean of theta at steps {160,170,180,190,200}  (the spec)
#   tailavg_avg50dense          mean of theta over the last 50 checkpoints
#   shrink_lam{0.25,0.5,0.75}   pure-shrinkage control theta_0 + lambda*(theta_final-theta_0)
#
# Read: averaging counts as having extracted a direction only if it beats the shrinkage
# family AT MATCHED KL. If the two lie on one curve, the gain is just "step back a bit".
#
# Arms run best-candidate-first: N=30 and N=20 are the only arms whose TRAIN fitness rose
# significantly (t=2.5/2.1), so they are where a coherent drift could exist; N=8/N=4 are
# the high-drift damaged arms, where shrinkage alone should dominate.
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
SRC=${SRC:-track_b/results/exp3b_math}
OUT=track_b/results/phase1_tailavg; LOG=track_b/logs/phase1_tailavg
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/driver.log"; }

log "=== Phase 1 tail-averaging START (model=$MODEL) ==="
for N in 30 20 8 4; do
  tag="fulles_N${N}"
  if [ ! -f "$SRC/${tag}.jsonl" ]; then log "$tag SKIP (no jsonl)"; continue; fi
  log "$tag start"
  $P4_PY track_b/src/tail_average.py \
    --jsonl "$SRC/${tag}.jsonl" --model "$MODEL" \
    --alpha 5e-4 --pop_seed 0 \
    --tail 160,170,180,190,200 --tail_dense 50 \
    --lambdas 0.25,0.5,0.75 \
    --eval_cap 300 --gpu $GPU \
    --out "$OUT/${tag}" > "$LOG/${tag}.log" 2>&1 \
    && log "$tag DONE" || log "$tag FAIL rc=$?"
done
log "=== Phase 1 tail-averaging ALL DONE ==="
