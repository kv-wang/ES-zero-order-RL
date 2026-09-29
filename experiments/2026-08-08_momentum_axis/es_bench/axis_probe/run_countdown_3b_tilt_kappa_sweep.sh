#!/usr/bin/env bash
# Countdown on Qwen2.5-3B-Instruct: sequential tilt kappa sweep (SINGLE GPU).
# Based on run_countdown_3b.sh; runs tilt for each initial kappa in order, beta fixed.
# DEFAULT: cosine-decay λ from κ0 → κ_end=0 over num_steps (see --tilt_lambda_cosine).
#
#   stage 0  untrained base reference (once)
#   stage 6  ES tilt for each kappa in KAPPA_LIST (default: 0.05 then 0.01 as λ_max)
#   stage 4  readout over all summaries under RES/
#
# Usage:  DRY_RUN=1 ./run_countdown_3b_tilt_kappa_sweep.sh
#         SMOKE=1   ./run_countdown_3b_tilt_kappa_sweep.sh
#         ./run_countdown_3b_tilt_kappa_sweep.sh
#         KAPPA_LIST="0.05 0.01" TILT_MOM_BETA=0.9 ./run_countdown_3b_tilt_kappa_sweep.sh
#         TILT_LAMBDA_COSINE=0 ./run_countdown_3b_tilt_kappa_sweep.sh   # fixed κ (old behavior)
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

STAGES="${STAGES:-0 6 4}"
GPU="${GPU:-0}"
TP="${TP:-1}"
SEEDS="${SEEDS:-0}"
N="${N:-30}"
B="${B:-100}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MAXTOK="${MAXTOK:-2048}"
EVAL_CAP="${EVAL_CAP:-300}"
SIGMA="${SIGMA:-1e-3}"; ALPHA="${ALPHA:-5e-4}"
MINI_BATCH="${MINI_BATCH:-64}"
KAPPA_LIST="${KAPPA_LIST:-0.2}"   # initial κ (λ_max) for each sequential run
TILT_KAPPA_END="${TILT_KAPPA_END:-0.0}" # final κ after cosine (default: anneal to vanilla)
TILT_LAMBDA_COSINE="${TILT_LAMBDA_COSINE:-1}"  # 1 = cosine-decay λ over steps (default ON)
TILT_MOM_BETA="${TILT_MOM_BETA:-0.9}"   # fixed EMA decay
TILT_WARMUP="${TILT_WARMUP:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MEM_POLL="${MEM_POLL:-10}"
RHO_SPLITS="${RHO_SPLITS:-20}"
S_PER_GEN="${S_PER_GEN:-0.18}"
ES_MEM_UTIL="${ES_MEM_UTIL:-0.70}"

TAG="countdown_3b_N${N}_B${B}"
if [[ "$SMOKE" == "1" ]]; then
  STEPS=2; EVAL_CAP=50; EVAL_EVERY=1; TAG="${TAG}_smoke"
fi

PER_STEP=$((N * B))
ES_GENERATIONS=$((STEPS * PER_STEP))
N_KAPPA=$(echo $KAPPA_LIST | wc -w)

RES="axis_probe/results/${TAG}"
LOGS="axis_probe/logs/${TAG}"
CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"
mkdir -p "$RES" "$LOGS" "$CKPT"
DRIVER="$LOGS/driver_tilt_kappa_sweep.log"
MEMCSV="$LOGS/gpu_mem_tilt_kappa_sweep.csv"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

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

plan() {
  local es_h tot ev_h one_h
  es_h=$(awk -v g="$ES_GENERATIONS" -v s="$S_PER_GEN" 'BEGIN{printf "%.1f", g*s/3600}')
  ev_h=$(awk -v k="$((STEPS / EVAL_EVERY + 1))" -v c="$EVAL_CAP" -v s="$S_PER_GEN" \
         'BEGIN{printf "%.1f", k*7*c*s/3600}')
  one_h=$(awk -v a="$es_h" -v e="$ev_h" 'BEGIN{printf "%.1f", a+e}')
  tot=$(awk -v one="$one_h" -v nk="$N_KAPPA" -v ns="$(echo $SEEDS | wc -w)" \
        'BEGIN{printf "%.1f", ns*nk*one}')
  cat <<EOF | tee -a "$DRIVER"
================ countdown @ Qwen2.5-3B-Instruct (tilt kappa sweep) ================
model            : $MODEL
stages           : $STAGES        seeds: $SEEDS        smoke: $SMOKE
GPU              : $GPU (single card, TP=$TP)
ES               : N=$N  B=$B  steps=$STEPS  sigma=$SIGMA  alpha=$ALPHA  mini_batch=$MINI_BATCH
tilt sweep       : kappa_list=[$KAPPA_LIST]  (sequential)
                   kappa_end=$TILT_KAPPA_END  lambda_cosine=$TILT_LAMBDA_COSINE
                   mom_beta=$TILT_MOM_BETA (fixed)  warmup=$TILT_WARMUP
                   Covariance: C = σ²(I + λ²·m̂·m̂ᵀ) where λ=sqrt(κ·d/(1-κ))
                   If lambda_cosine=1: each run cosine-decays λ from κ to κ_end.
ES generations   : $STEPS x $N x $B = $ES_GENERATIONS  per kappa
eval             : every $EVAL_EVERY updates + final, cap=$EVAL_CAP, max_tokens=$MAXTOK
rho              : offline from correct_bits, $RHO_SPLITS splits/step
vLLM mem util    : $ES_MEM_UTIL
projection       : ~${one_h}h per (seed,kappa) x ${N_KAPPA} kappas x $(echo $SEEDS | wc -w) seed(s)
                   -> ~${tot}h total at ${S_PER_GEN}s/gen
results          : $RES
checkpoints      : $CKPT
driver log       : $DRIVER
===================================================================================
EOF
}
plan
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> stopping before any GPU work."; exit 0; fi
preflight || { say "ABORT on preflight. Override with SKIP_PREFLIGHT=1 if you know better."; \
               [[ "${SKIP_PREFLIGHT:-0}" == "1" ]] || exit 3; }

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
  echo "{\"stage\":\"$label\",\"rc\":$rc,\"seconds\":$((t1-t0))}" >> "$LOGS/stage_times_tilt_kappa_sweep.jsonl"
  phase "idle"
  return $rc
}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ---- stage 0: base reference (once) ----
if has_stage 0; then
  out="$RES/base_summary.json"
  if [[ -s "$out" ]]; then
    say "STAGE0 skip (base summary exists)"
  else
    run_stage "base" "$LOGS/base_tilt_kappa_sweep.log" \
      "$P4_PY" axis_probe/eval_base.py \
        --out_prefix "$RES/base" --max_tokens "$MAXTOK" \
        --eval_cap "$EVAL_CAP" --gpu "$GPU" --tensor_parallel_size "$TP" \
        --gpu_mem_util "$ES_MEM_UTIL" --include_countdown
    [[ $? -eq 0 ]] || { say "!! base eval FAILED, aborting."; exit 1; }
  fi
fi

# ---- stage 6: tilt for each kappa, sequentially ----
if has_stage 6; then
  for kappa in $KAPPA_LIST; do
    say "==== tilt kappa=$kappa  mom_beta=$TILT_MOM_BETA  cosine=$TILT_LAMBDA_COSINE ===="
    for seed in $SEEDS; do
      if [[ "$TILT_LAMBDA_COSINE" == "1" ]]; then
        prefix="tilt_N${N}_B${B}_kappa${kappa}_to${TILT_KAPPA_END}_cos_s${seed}"
      else
        prefix="tilt_N${N}_B${B}_kappa${kappa}_s${seed}"
      fi
      jsonl="$RES/${prefix}.jsonl"
      summary="$RES/${prefix}_summary.json"

      if [[ -s "$summary" ]]; then
        say "STAGE6 skip tilt kappa=$kappa seed=$seed (summary exists)"
      else
        TILT_EXTRA=()
        if [[ "$TILT_LAMBDA_COSINE" == "1" ]]; then
          TILT_EXTRA+=(--tilt_lambda_cosine --tilt_kappa_end "$TILT_KAPPA_END")
        fi
        run_stage "tilt_k${kappa}_s${seed}" "$LOGS/${prefix}.log" \
          "$P4_PY" axis_probe/src/es_train_axis.py \
            --dataset countdown --variant tilt \
            --pop_seed "$seed" \
            --population_size "$N" --batch "$B" --num_steps "$STEPS" \
            --sigma "$SIGMA" --alpha "$ALPHA" --max_tokens "$MAXTOK" \
            --tilt_kappa "$kappa" --tilt_mom_beta "$TILT_MOM_BETA" \
            --tilt_warmup "$TILT_WARMUP" \
            "${TILT_EXTRA[@]}" \
            --eval_interval "$EVAL_EVERY" --eval_cap "$EVAL_CAP" --eval_final \
            --mini_batch "$MINI_BATCH" --gpu "$GPU" --tensor_parallel_size "$TP" \
            --gpu_mem_util "$ES_MEM_UTIL" \
            --out_prefix "$RES/${prefix}"
        [[ $? -eq 0 ]] || { say "!! tilt kappa=$kappa seed=$seed FAILED, aborting."; exit 1; }

        run_stage "rho_tilt_k${kappa}_s${seed}" "$LOGS/${prefix}_rho.log" \
          "$P4_PY" axis_probe/rho_curve.py --jsonl "$jsonl" --splits "$RHO_SPLITS"
      fi
    done
  done
fi

# ---- stage 4: readout ----
if has_stage 4; then
  say "STAGE4 readout -> $RES/REPORT_tilt_kappa_sweep.md"
  "$P4_PY" axis_probe/report_countdown_paperB.py \
      --results "$RES" --logs "$LOGS" --mem_csv "$MEMCSV" \
      --stage_times "$LOGS/stage_times_tilt_kappa_sweep.jsonl" \
      --es_generations "$ES_GENERATIONS" --grpo_rollouts 0 \
      --n "$N" --b "$B" --steps "$STEPS" \
      --grpo_steps 0 --grpo_rollouts_per_step 0 \
      --es_mem_util "$ES_MEM_UTIL" --grpo_mem_util 0 \
      --out "$RES/REPORT_tilt_kappa_sweep.md" | tee -a "$DRIVER"
  say "ALL DONE. Results in $RES/"
  say "  kappa_list: $KAPPA_LIST  beta=$TILT_MOM_BETA"
  say "  tilt summaries: $(ls $RES/tilt_*_summary.json 2>/dev/null | wc -l)"
fi
