#!/usr/bin/env bash
# Countdown line at the paper's batch size: ES B=1000 (the full-ish training pool) instead of
# the 08-16 run's B=100. MODIFIED 2026-08-21: STEPS=100 (3M generations), dual-GPU parallel.
#
#   stage 0  untrained base reference  (countdown ID + math battery as OOD)
#   stage 2  ES vanilla on GPU0        -\ es_train_axis.py --dataset countdown --batch $B
#   stage 3  ES baseaxis on GPU1       -/  (parallel)
#   stage 4  readout: accuracy + wall clock + whole-card peak GPU memory per method
#
# WHY B=1000
# ----------
# PAPER_FIDELITY_AUDIT.md: the paper scores each ES member on the entire training set (the
# Sudoku section states 800 explicitly); this repo defaulted to B=8, which leaves fitness with
# only ~3 distinct values and split-half rho == 0.000 on every math arm ever measured. The 08-16
# countdown run used B=100. countdown.json has 2200 rows with [:300] pinned for eval, so the
# train pool is 1900 and B=1000 is the largest round batch that still fits correct_bits (one
# Python int per member, B<=1024).
#
# DEVIATION from the paper, deliberate, state it in any writeup: the trainer draws B indices per
# step WITH REPLACEMENT from the 1900-row pool and RESAMPLES every step (es_train_axis.py
# step_batches). The paper holds one fixed full-set batch, so its objective is stationary while
# this one still moves between steps. B=1000 shrinks the per-step noise ~10x versus 08-16 but
# does not remove it.
#
# COST -- READ BEFORE LAUNCHING
# -----------------------------
# 08-16 measured 217.3 s/step at N=30 B=100 on this exact model (= 0.0724 s/generation). Cost is
# linear in STEPS*N*B, so B=1000 costs 10x per step. Modified defaults (N=30 B=1000 STEPS=100 =
# 3M generations/arm) project to roughly:
#     vanilla ~60 h  +  baseaxis ~60 h  = ~120 h SERIAL, or ~60 h PARALLEL on 2 GPUs
# Run DRY_RUN=1 first: it prints the resolved shape and projection, then exits without GPU work.
#
# Usage:  DRY_RUN=1 ./run_countdown_paperB.sh          # plan only, no GPU
#         ./run_countdown_paperB.sh                    # full run, single seed, parallel
#         STEPS=50 ./run_countdown_paperB.sh           # 1.5M gens, half the default
#         STAGES="0 2 4" ./run_countdown_paperB.sh     # base + vanilla + readout (skip baseaxis)
#         SEEDS="0 1" ./run_countdown_paperB.sh        # two seeds (doubles the ES cost)
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

# 1.5B-Instruct, not the 3B the other countdown driver defaults to: this run is meant to be read
# against 08-16, which is 1.5B-Instruct. Every downstream path keys off MODEL via
# config.RESULTS_SUFFIX, so setting it here is what scopes the results dirs.
export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

STAGES="${STAGES:-0 2 3 4}"
GPU="${GPU:-0,1}"
TP="${TP:-2}"
SEEDS="${SEEDS:-0}"
N="${N:-30}"
B="${B:-1000}"                # the point of this script
MAXTOK="${MAXTOK:-2048}"      # 07-30 protocol: SAME cap for training and eval, never mixed
EVAL_CAP="${EVAL_CAP:-300}"
SIGMA="${SIGMA:-1e-3}"; ALPHA="${ALPHA:-5e-4}"    # paper's reference scale, and alpha = sigma/2
MINI_BATCH="${MINI_BATCH:-64}"  # vLLM generate chunk; bounds peak KV so B=1000 does not blow up
DRY_RUN="${DRY_RUN:-0}"
MEM_POLL="${MEM_POLL:-10}"      # seconds between nvidia-smi samples

# vLLM KV budget for ES (no GRPO in this run)
ES_MEM_UTIL="${ES_MEM_UTIL:-0.85}"

# ---- shape resolution: STEPS wins, else GEN_BUDGET, else the default ----
PER_STEP=$((N * B))
if [[ -n "${STEPS:-}" ]]; then
  :
elif [[ -n "${GEN_BUDGET:-}" ]]; then
  STEPS=$((GEN_BUDGET / PER_STEP))
  [[ $STEPS -lt 1 ]] && { echo "GEN_BUDGET=$GEN_BUDGET < one step ($PER_STEP gens)"; exit 2; }
else
  STEPS=100
fi
ES_GENERATIONS=$((STEPS * PER_STEP))
# Results/logs are scoped by model AND by B. The other drivers key only on model, so pointing
# this run at the usual dir would let "summary already exists -> skip" hand back 08-16's B=100
# numbers as if they were B=1000 (the exact failure documented in the 08-14 ledger row). 08-16's
# artefacts are never written to by this script.
TAG="countdown_paperB${B}_N${N}"
RES="axis_probe/results/${TAG}"
LOGS="axis_probe/logs/${TAG}"
CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"    # NOT /tmp: the 08-19 run lost its checkpoints
mkdir -p "$RES" "$LOGS" "$CKPT"
DRIVER="$LOGS/driver.log"
MEMCSV="$LOGS/gpu_mem.csv"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

# ---- plan ----
# 0.0724 s/generation for ES is the 08-16 measurement on this model (217.3 s/step / 3000 gens).
# Same model, same cap -> usable as a projection, not as a promise: B=1000 changes the vLLM batch shape.
plan() {
  local es_h tot
  es_h=$(awk -v g="$ES_GENERATIONS" 'BEGIN{printf "%.1f", g*0.0724/3600}')
  tot=$(awk -v a="$es_h" -v s="$(echo $SEEDS | wc -w)" -v n=2 'BEGIN{printf "%.1f", s*n*a}')
  cat <<EOF | tee -a "$DRIVER"
================ countdown @ paper-scale B ================
model            : $MODEL
stages           : $STAGES        seeds: $SEEDS
GPUs             : $GPU (TP=$TP, serial: vanilla → baseaxis)
ES               : N=$N  B=$B  steps=$STEPS  sigma=$SIGMA  alpha=$ALPHA  mini_batch=$MINI_BATCH
ES generations   : $STEPS x $N x $B = $ES_GENERATIONS   (08-16 was N=30 B=100 steps=100 = 300000)
max_tokens       : $MAXTOK  (train == eval)      eval_cap: $EVAL_CAP
vLLM mem util    : ES $ES_MEM_UTIL
projection       : ES ~${es_h}h/arm x 2 variants (serial) -> ~${tot}h total
results          : $RES
checkpoints      : $CKPT
===========================================================
EOF
}
plan
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> stopping before any GPU work."; exit 0; fi

# ---- whole-card memory sampler ----
# MEMORY_REPORT.md (08-13) established the only valid measurement here: torch's
# max_memory_allocated is PER PROCESS, so it misses verl's vLLM entirely (separate pid) while
# capturing all of ES's KV (same process, forced by collective_rpc). Comparing those two numbers
# is meaningless. nvidia-smi sees the whole card for every arm, so that is what is reported.
# For multi-GPU tensor parallelism: sum memory across all GPUs.
MEMCSV="$LOGS/gpu_mem.csv"
echo "phase,timestamp,mem_used_mib" > "$MEMCSV"
echo "idle" > "$LOGS/.phase"

( while true; do
    IFS=',' read -ra GPUS <<< "$GPU"
    total_mem=0
    for g in "${GPUS[@]}"; do
      mem=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      total_mem=$((total_mem + ${mem:-0}))
    done
    p=$(cat "$LOGS/.phase" 2>/dev/null || echo unknown)
    [[ $total_mem -gt 0 ]] && echo "$p,$(date +%s),$total_mem" >> "$MEMCSV"
    sleep "$MEM_POLL"
  done ) &
SAMPLER=$!

trap 'kill $SAMPLER 2>/dev/null' EXIT
phase() { echo "$1" > "$LOGS/.phase"; }

# rc capture: `echo "rc=$?"` after a command reports the exit code of the LAST expansion in that
# echo, not of the command -- a $(date) in the same line resets it to 0. That bug made every
# failed GRPO stage report rc=0 COMPLETE in the 08-15 run. Capture first, then log.
run_stage() {
  local label="$1" log="$2"; shift 2
  phase "$label"
  say "START $label -> $log"
  local t0 rc t1
  t0=$(date +%s)
  "$@" > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  say "END   $label rc=$rc elapsed=$((t1 - t0))s"
  echo "{\"stage\":\"$label\",\"rc\":$rc,\"seconds\":$((t1-t0))}" >> "$LOGS/stage_times.jsonl"
  phase "idle"
  return $rc
}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# Mirror of run_grpo.sh's own naming: OUT=results/grpo$SUF, TAG=grpo_${TASK}_${BUDGET}${STEPS}${SUF}.
# SUF comes from config.RESULTS_SUFFIX for the MODEL exported above, so ask config rather than
# hardcoding it -- that is what keeps this pointing at the right file when MODEL changes.
SUF=$("$P4_PY" -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(C.RESULTS_SUFFIX)")

# ---- stage 0: untrained base reference ----
# include_countdown makes countdown the ID metric; the six math sets come along as OOD. Without
# this stage there is no "did training help at all" baseline -- the gap ES_VS_BASE_HISTORY.md
# found in three of the four July trys.
if has_stage 0; then
  if [[ -f "$RES/base_summary.json" ]]; then
    say "SKIP stage0 base: $RES/base_summary.json exists"
  else
    run_stage "base" "$LOGS/base.log" \
      "$P4_PY" axis_probe/eval_base.py \
        --out_prefix "$RES/base" --include_countdown \
        --eval_cap "$EVAL_CAP" --max_tokens "$MAXTOK" --gpu 0 \
      || say "WARN stage0 failed; later stages still run but lose their reference"
  fi
fi

# ---- stages 2/3: ES vanilla and baseaxis (SERIAL, both using dual-GPU TP) ----
# Same trainer, same data/reward/eval path, only --variant differs; baseaxis additionally holds a
# theta0 snapshot (08-16 measured ~9 GB more peak for it at N=30).
es_arm() {
  local variant="$1" seed="$2"
  local pref="$RES/${variant}_N${N}_B${B}_s${seed}"
  if [[ -f "${pref}_summary.json" ]]; then
    say "SKIP ${variant} s${seed}: ${pref}_summary.json exists"
    return 0
  fi
  run_stage "es_${variant}_s${seed}" "$LOGS/${variant}_s${seed}.log" \
    "$P4_PY" axis_probe/src/es_train_axis.py \
      --variant "$variant" --dataset countdown \
      --population_size "$N" --batch "$B" --mini_batch "$MINI_BATCH" \
      --num_steps "$STEPS" --pop_seed "$seed" \
      --sigma "$SIGMA" --alpha "$ALPHA" --max_tokens "$MAXTOK" \
      --gpu "$GPU" --tensor_parallel_size "$TP" \
      --gpu_mem_util "$ES_MEM_UTIL" \
      --eval_cap "$EVAL_CAP" --eval_final --eval_interval 10 --kl \
      --out_prefix "$pref" \
    || say "WARN ${variant} s${seed} failed -- see $LOGS/${variant}_s${seed}.log"
}

for s in $SEEDS; do
  has_stage 2 && es_arm vanilla "$s"
  has_stage 3 && es_arm baseaxis "$s"
done

# ---- stage 4: readout ----
phase "idle"
if has_stage 4; then
  say "START readout"
  "$P4_PY" axis_probe/report_countdown_paperB.py \
      --results "$RES" --logs "$LOGS" --mem_csv "$MEMCSV" \
      --stage_times "$LOGS/stage_times.jsonl" \
      --es_generations "$ES_GENERATIONS" --grpo_rollouts 0 \
      --n "$N" --b "$B" --steps "$STEPS" \
      --grpo_steps 0 --grpo_rollouts_per_step 0 \
      --es_mem_util "$ES_MEM_UTIL" --grpo_mem_util 0 \
      --out "$RES/REPORT.md" | tee -a "$DRIVER"
  say "END   readout -> $RES/REPORT.md"
fi

kill $SAMPLER 2>/dev/null
say "ALL DONE. results=$RES logs=$LOGS"
