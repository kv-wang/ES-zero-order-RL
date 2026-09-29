#!/usr/bin/env bash
# Countdown with rank-one momentum tilt ES variant
#
# Tilt variant applies tilted Gaussian perturbations to ALL N members:
#   ε_i = σ·ξ_i + λσ·m̂·ζ_i where ξ_i~N(0,I), ζ_i~N(0,1), m̂ is unit momentum direction
#   Covariance: C = σ²(I + λ²·m̂·m̂ᵀ) creates prolate spheroid around momentum axis
#   Energy fraction κ (kappa): dimensionless, fraction of energy in m̂ direction
#   Lambda: λ = sqrt(κ·d/(1-κ)) where d is parameter count
#
# Usage:  DRY_RUN=1 ./run_countdown_tilt.sh          # plan only, no GPU
#         ./run_countdown_tilt.sh                    # full run with default kappa=0.2
#         KAPPA=0.1 ./run_countdown_tilt.sh          # sweep kappa values
#         KAPPA=0.0 ./run_countdown_tilt.sh          # vanilla ES (ablation baseline)
set -uo pipefail
cd "$(dirname "$0")/.."
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

STAGES="${STAGES:-0 2 4}"
GPU="${GPU:-0}"
TP="${TP:-1}"
SEEDS="${SEEDS:-0}"
N="${N:-30}"
B="${B:-100}"
MAXTOK="${MAXTOK:-2048}"
EVAL_CAP="${EVAL_CAP:-300}"
SIGMA="${SIGMA:-1e-3}"
ALPHA="${ALPHA:-5e-4}"
MINI_BATCH="${MINI_BATCH:-64}"
DRY_RUN="${DRY_RUN:-0}"
MEM_POLL="${MEM_POLL:-10}"

# Tilt-specific parameters
KAPPA="${KAPPA:-0.2}"           # energy fraction in momentum direction (0=vanilla, 0.2 typical)
MOM_BETA="${MOM_BETA:-0.9}"     # EMA decay for momentum buffer
TILT_WARMUP="${TILT_WARMUP:-1}" # steps of vanilla ES before tilt activates

ES_MEM_UTIL="${ES_MEM_UTIL:-0.85}"

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

TAG="countdown_tilt_k${KAPPA}_N${N}_B${B}"
RES="axis_probe/results/${TAG}"
LOGS="axis_probe/logs/${TAG}"
CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"
mkdir -p "$RES" "$LOGS" "$CKPT"
DRIVER="$LOGS/driver.log"
MEMCSV="$LOGS/gpu_mem.csv"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

plan() {
  local es_h tot
  es_h=$(awk -v g="$ES_GENERATIONS" 'BEGIN{printf "%.1f", g*0.0724/3600}')
  tot=$(awk -v a="$es_h" -v s="$(echo $SEEDS | wc -w)" 'BEGIN{printf "%.1f", s*a}')
  cat <<EOF | tee -a "$DRIVER"
================ countdown tilt variant ================
model            : $MODEL
stages           : $STAGES        seeds: $SEEDS
GPUs             : $GPU (TP=$TP)
ES               : N=$N  B=$B  steps=$STEPS  sigma=$SIGMA  alpha=$ALPHA  mini_batch=$MINI_BATCH
Tilt params      : kappa=$KAPPA  mom_beta=$MOM_BETA  warmup=$TILT_WARMUP
ES generations   : $STEPS x $N x $B = $ES_GENERATIONS
max_tokens       : $MAXTOK  (train == eval)      eval_cap: $EVAL_CAP
vLLM mem util    : ES $ES_MEM_UTIL
projection       : ES ~${es_h}h x $(echo $SEEDS | wc -w) seed(s) -> ~${tot}h total
results          : $RES
checkpoints      : $CKPT
========================================================
EOF
}
plan
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> stopping before any GPU work."; exit 0; fi

echo "phase,timestamp,mem_used_mib" > "$MEMCSV"
echo "idle" > "$LOGS/.phase"

( while true; do
    IFS=',' read -ra GPUS <<< "$GPU"
    total_mem=0
    for g in "${GPUS[@]}"; do
      mem=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      total_mem=$((total_mem + ${mem:-0}))
    done
    ts=$(date +%s)
    phase=$(cat "$LOGS/.phase" 2>/dev/null || echo "unknown")
    echo "$phase,$ts,$total_mem" >> "$MEMCSV"
    sleep "$MEM_POLL"
  done
) &
MEM_PID=$!
trap "kill $MEM_PID 2>/dev/null" EXIT

for SEED in $SEEDS; do
  say "seed $SEED: stages $STAGES"

  for stage in $STAGES; do
    case "$stage" in
      0)
        say "stage 0: base eval"
        echo "eval_base" > "$LOGS/.phase"
        python axis_probe/eval_base.py \
          --dataset countdown \
          --eval_cap "$EVAL_CAP" \
          --max_tokens "$MAXTOK" \
          --gpu_memory_utilization "$ES_MEM_UTIL" \
          --tensor_parallel_size "$TP" 2>&1 | tee "$LOGS/eval_base_s${SEED}.log"
        ;;
      2)
        say "stage 2: ES tilt (kappa=$KAPPA) seed $SEED"
        echo "es_tilt_s${SEED}" > "$LOGS/.phase"
        VARIANT_LOG="$LOGS/tilt_s${SEED}.log"
        CUDA_VISIBLE_DEVICES="$GPU" python axis_probe/src/es_train_axis.py \
          --variant tilt \
          --dataset countdown \
          --batch_size "$B" \
          --num_steps "$STEPS" \
          --population_size "$N" \
          --sigma "$SIGMA" \
          --alpha "$ALPHA" \
          --seed "$SEED" \
          --max_tokens "$MAXTOK" \
          --mini_batch "$MINI_BATCH" \
          --gpu_memory_utilization "$ES_MEM_UTIL" \
          --tensor_parallel_size "$TP" \
          --tilt_kappa "$KAPPA" \
          --tilt_mom_beta "$MOM_BETA" \
          --tilt_warmup "$TILT_WARMUP" \
          --checkpoint_dir "$CKPT/tilt_s${SEED}" 2>&1 | tee "$VARIANT_LOG"
        ;;
      4)
        say "stage 4: readout"
        echo "readout" > "$LOGS/.phase"
        python axis_probe/report_countdown_paperB.py \
          --tag "$TAG" \
          --seeds "$SEED" 2>&1 | tee "$LOGS/report_s${SEED}.log"
        ;;
      *)
        say "unknown stage $stage, skipping"
        ;;
    esac
  done
done

say "done"
echo "idle" > "$LOGS/.phase"
sleep 2
kill $MEM_PID 2>/dev/null
wait $MEM_PID 2>/dev/null || true
