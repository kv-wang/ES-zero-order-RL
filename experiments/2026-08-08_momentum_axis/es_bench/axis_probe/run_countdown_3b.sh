#!/usr/bin/env bash
# Countdown on Qwen2.5-3B-Instruct: N=30, B=100, steps=100, SINGLE GPU.
# TILT HALF→VANILLA (default): first TILT_SWITCH_FRAC steps use κ=TILT_KAPPA tilt,
# remaining steps use vanilla ES (κ=0).
#
#   stage 0  untrained base reference   (countdown ID + the six math sets as OOD)
#   stage 6  ES tilt with piecewise half schedule
#   stage 4  readout: accuracy + wall clock + whole-card peak GPU memory per method
#
# WHY 3B
# ------
# 7B GRPO failed on single GPU even at bs=4 (139.15 GiB peak > 143.8 GiB capacity). 3B should fit
# comfortably in single-GPU memory for ES training. This script matches the 7B countdown protocol
# (N=30, B=100, steps=100, eval_every=10) but runs on a single GPU without tensor parallelism.
#
# MEMORY BUDGET (single GPU, 143771 MiB available)
# ------------------------------------------------
#   vLLM fp16 weights (no TP) ................ ~6 GB   (3B bf16 model)
#   es_worker._es_base, fp32 ................. ~12 GB   (fp32 snapshot, see es_worker.py:19-26)
#   tilt momentum buffer, fp32 ............... ~12 GB
#   vLLM budget at util=0.70 ................. ~100 GB  (weights + KV)
#   tilt worst case .......................... ~100 + 24 = 124 GB  (86% of card, safe margin)
#
# Usage:  DRY_RUN=1 ./run_countdown_3b.sh        # plan only, no GPU
#         SMOKE=1   ./run_countdown_3b.sh        # 2 steps, cap 50, calibrates s/step
#         ./run_countdown_3b.sh                  # full run: half tilt κ=0.2 then vanilla
#         TILT_HALF_VANILLA=0 ./run_countdown_3b.sh   # fixed-κ tilt (old default)
#         STAGES="0 6 4" ./run_countdown_3b.sh   # base + tilt + readout
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

STAGES="${STAGES:-0 6 4}"
GPU="${GPU:-0}"          # Single GPU
TP="${TP:-1}"            # No tensor parallelism
SEEDS="${SEEDS:-0}"
N="${N:-30}"
B="${B:-100}"
STEPS="${STEPS:-100}"
# Must DIVIDE STEPS. es_train_axis.py gates the curve on (step+1) % EVAL_EVERY, so the points land
# on updates EVAL_EVERY, 2*EVAL_EVERY, ..., STEPS; a non-divisor silently drops the end state.
# The arms still pass --eval_final even though the curve now reaches STEPS: report_countdown_paperB.py
# reads accuracy ONLY from summary["eval_final"], so without it every cell in REPORT.md is `--`.
# It re-measures the same theta for ~80 s, which doubles as a check on vLLM's run-to-run spread.
EVAL_EVERY="${EVAL_EVERY:-10}"
MAXTOK="${MAXTOK:-2048}"
EVAL_CAP="${EVAL_CAP:-300}"
SIGMA="${SIGMA:-1e-3}"; ALPHA="${ALPHA:-5e-4}"
MINI_BATCH="${MINI_BATCH:-64}"
TILT_KAPPA="${TILT_KAPPA:-0.2}"        # κ for the tilt phase (first switch_frac of steps)
TILT_KAPPA_END="${TILT_KAPPA_END:-0.0}"  # Final κ when TILT_LAMBDA_COSINE=1
TILT_LAMBDA_COSINE="${TILT_LAMBDA_COSINE:-0}"  # 1 = cosine-decay λ (off by default)
TILT_HALF_VANILLA="${TILT_HALF_VANILLA:-1}"    # 1 = first half tilt, second half vanilla (DEFAULT)
TILT_SWITCH_FRAC="${TILT_SWITCH_FRAC:-0.5}"    # fraction of steps that use tilt before switch
TILT_MOM_BETA="${TILT_MOM_BETA:-0.9}"  # EMA decay for momentum
TILT_WARMUP="${TILT_WARMUP:-1}"        # Steps before tilt activates
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MEM_POLL="${MEM_POLL:-10}"
RHO_SPLITS="${RHO_SPLITS:-20}"
S_PER_GEN="${S_PER_GEN:-0.18}"  # Measured on the 08-27 3B vanilla arm (538.2 s/step at N=30 B=100)

ES_MEM_UTIL="${ES_MEM_UTIL:-0.70}"   # Single GPU, can use more

if [[ "$TILT_LAMBDA_COSINE" == "1" && "$TILT_HALF_VANILLA" == "1" ]]; then
  echo "ERROR: set only one of TILT_LAMBDA_COSINE=1 and TILT_HALF_VANILLA=1" >&2
  exit 2
fi

TAG="countdown_3b_N${N}_B${B}"
if [[ "$SMOKE" == "1" ]]; then
  STEPS=2; EVAL_CAP=50; EVAL_EVERY=1; TAG="${TAG}_smoke"
fi

PER_STEP=$((N * B))
ES_GENERATIONS=$((STEPS * PER_STEP))

RES="axis_probe/results/${TAG}"
LOGS="axis_probe/logs/${TAG}"
CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"
mkdir -p "$RES" "$LOGS" "$CKPT"
DRIVER="$LOGS/driver.log"
MEMCSV="$LOGS/gpu_mem.csv"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

# ---- preflight ----
preflight() {
  local bad=0
  local snap
  snap=$(HF_HUB_OFFLINE=1 "$P4_PY" - <<'PY' 2>/dev/null
try:
    from huggingface_hub import snapshot_download
    import glob, os
    p = snapshot_download(os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct"), local_files_only=True)
    n = len(glob.glob(os.path.join(p, "*.safetensors")))
    gb = sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(os.path.join(p, "*.safetensors")))/2**30
    print(f"{p}|{n}|{gb:.2f}")
except Exception as e:
    print(f"ERR|{e}")
PY
)
  if [[ "$snap" == ERR* || -z "$snap" ]]; then
    say "PREFLIGHT FAIL: $MODEL does not resolve offline ($snap)"; bad=1
  else
    say "PREFLIGHT model: $snap"
  fi

  local used
  used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  say "PREFLIGHT gpu$GPU: ${used:-?} MiB in use"
  if [[ -n "${used:-}" && $used -gt 5000 ]]; then
    say "PREFLIGHT FAIL: gpu$GPU has ${used} MiB in use; free it first"; bad=1
  fi

  return $bad
}

# ---- plan ----
plan() {
  local es_h tot ev_h
  es_h=$(awk -v g="$ES_GENERATIONS" -v s="$S_PER_GEN" 'BEGIN{printf "%.1f", g*s/3600}')
  ev_h=$(awk -v k="$((STEPS / EVAL_EVERY + 1))" -v c="$EVAL_CAP" -v s="$S_PER_GEN" \
         'BEGIN{printf "%.1f", k*7*c*s/3600}')
  tot=$(awk -v a="$es_h" -v e="$ev_h" -v s="$(echo $SEEDS | wc -w)" 'BEGIN{printf "%.1f", s*(a+e)}')
  cat <<EOF | tee -a "$DRIVER"
================ countdown @ Qwen2.5-3B-Instruct (tilt half→vanilla) ================
model            : $MODEL
stages           : $STAGES        seeds: $SEEDS        smoke: $SMOKE
GPU              : $GPU (single card, TP=$TP)
ES               : N=$N  B=$B  steps=$STEPS  sigma=$SIGMA  alpha=$ALPHA  mini_batch=$MINI_BATCH
tilt             : kappa=$TILT_KAPPA  kappa_end=$TILT_KAPPA_END  lambda_cosine=$TILT_LAMBDA_COSINE
                   half_vanilla=$TILT_HALF_VANILLA  switch_frac=$TILT_SWITCH_FRAC
                   mom_beta=$TILT_MOM_BETA  warmup=$TILT_WARMUP
                   Covariance: C = σ²(I + λ²·m̂·m̂ᵀ) where λ=sqrt(κ·d/(1-κ))
                   If lambda_cosine=1: λ cosine-decays κ→κ_end over steps (κ=λ²/(d+λ²)).
                   If half_vanilla=1: κ for first switch_frac of steps, then vanilla ES.
ES generations   : $STEPS x $N x $B = $ES_GENERATIONS
eval             : every $EVAL_EVERY updates + final, cap=$EVAL_CAP, max_tokens=$MAXTOK (train == eval)
rho              : offline from correct_bits, $RHO_SPLITS splits/step
vLLM mem util    : $ES_MEM_UTIL
projection       : train ~${es_h}h + eval ~${ev_h}h  ->  ~${tot}h total
                   ESTIMATE at ${S_PER_GEN}s/gen -- run SMOKE=1 first to calibrate.
results          : $RES
checkpoints      : $CKPT
==============================================================================
EOF
}
plan
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> stopping before any GPU work."; exit 0; fi
preflight || { say "ABORT on preflight. Override with SKIP_PREFLIGHT=1 if you know better."; \
               [[ "${SKIP_PREFLIGHT:-0}" == "1" ]] || exit 3; }

# ---- memory sampler ----
echo "phase,timestamp,mem_used_mib" > "$MEMCSV"
echo "idle" > "$LOGS/.phase"

( while true; do
    mem=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    p=$(cat "$LOGS/.phase" 2>/dev/null || echo unknown)
    [[ ${mem:-0} -gt 0 ]] && echo "$p,$(date +%s),${mem:-0}" >> "$MEMCSV"
    sleep "$MEM_POLL"
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
phase() { echo "$1" > "$LOGS/.phase"; }

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

# ---- stage 0: base reference ----
if has_stage 0; then
  for seed in $SEEDS; do
    out="$RES/base_summary.json"
    if [[ -s "$out" ]]; then
      say "STAGE0 skip (base summary exists)"
    else
      run_stage "base" "$LOGS/base.log" \
        "$P4_PY" axis_probe/eval_base.py \
          --out_prefix "$RES/base" --max_tokens "$MAXTOK" \
          --eval_cap "$EVAL_CAP" --gpu "$GPU" --tensor_parallel_size "$TP" \
          --gpu_mem_util "$ES_MEM_UTIL" --include_countdown
      [[ $? -eq 0 ]] || { say "!! base eval FAILED, aborting."; exit 1; }
    fi
    break  # base only needs to run once, not per seed
  done
fi

# ---- stage 6: tilt ----
if has_stage 6; then
  for seed in $SEEDS; do
    if [[ "$TILT_LAMBDA_COSINE" == "1" ]]; then
      prefix="tilt_N${N}_B${B}_kappa${TILT_KAPPA}_to${TILT_KAPPA_END}_cos_s${seed}"
    elif [[ "$TILT_HALF_VANILLA" == "1" ]]; then
      prefix="tilt_N${N}_B${B}_kappa${TILT_KAPPA}_half${TILT_SWITCH_FRAC}_s${seed}"
    else
      prefix="tilt_N${N}_B${B}_kappa${TILT_KAPPA}_s${seed}"
    fi
    jsonl="$RES/${prefix}.jsonl"
    summary="$RES/${prefix}_summary.json"

    if [[ -s "$summary" ]]; then
      say "STAGE6 skip tilt seed=$seed (summary exists)"
    else
      TILT_EXTRA=()
      if [[ "$TILT_LAMBDA_COSINE" == "1" ]]; then
        TILT_EXTRA+=(--tilt_lambda_cosine --tilt_kappa_end "$TILT_KAPPA_END")
      elif [[ "$TILT_HALF_VANILLA" == "1" ]]; then
        TILT_EXTRA+=(--tilt_half_vanilla --tilt_switch_frac "$TILT_SWITCH_FRAC")
      fi
      run_stage "tilt_s${seed}" "$LOGS/${prefix}.log" \
        "$P4_PY" axis_probe/src/es_train_axis.py \
          --dataset countdown --variant tilt \
          --pop_seed "$seed" \
          --population_size "$N" --batch "$B" --num_steps "$STEPS" \
          --sigma "$SIGMA" --alpha "$ALPHA" --max_tokens "$MAXTOK" \
          --tilt_kappa "$TILT_KAPPA" --tilt_mom_beta "$TILT_MOM_BETA" \
          --tilt_warmup "$TILT_WARMUP" \
          "${TILT_EXTRA[@]}" \
          --eval_interval "$EVAL_EVERY" --eval_cap "$EVAL_CAP" --eval_final \
          --mini_batch "$MINI_BATCH" --gpu "$GPU" --tensor_parallel_size "$TP" \
          --gpu_mem_util "$ES_MEM_UTIL" \
          --out_prefix "$RES/${prefix}"
      [[ $? -eq 0 ]] || { say "!! tilt seed=$seed FAILED, aborting."; exit 1; }

      # rho curve
      run_stage "rho_tilt_s${seed}" "$LOGS/${prefix}_rho.log" \
        "$P4_PY" axis_probe/rho_curve.py --jsonl "$jsonl" --splits "$RHO_SPLITS"
    fi
  done
fi

# ---- stage 4: readout ----
if has_stage 4; then
  say "STAGE4 readout -> $RES/REPORT.md"
  # GRPO args zeroed: this line has no GRPO arm. The reporter is shared with run_countdown_7b.sh
  # rather than forked, so the 3B and 7B tables stay column-identical and can be read side by side.
  # It writes --out itself and echoes the report to stdout; do NOT redirect stdout onto REPORT.md.
  "$P4_PY" axis_probe/report_countdown_paperB.py \
      --results "$RES" --logs "$LOGS" --mem_csv "$MEMCSV" \
      --stage_times "$LOGS/stage_times.jsonl" \
      --es_generations "$ES_GENERATIONS" --grpo_rollouts 0 \
      --n "$N" --b "$B" --steps "$STEPS" \
      --grpo_steps 0 --grpo_rollouts_per_step 0 \
      --es_mem_util "$ES_MEM_UTIL" --grpo_mem_util 0 \
      --out "$RES/REPORT.md" | tee -a "$DRIVER"
  say "ALL DONE. Results in $RES/"
  say "  base: $(ls $RES/base_*_summary.json 2>/dev/null | wc -l) seed(s)"
  say "  tilt: $(ls $RES/tilt_*_summary.json 2>/dev/null | wc -l) seed(s)"
fi
