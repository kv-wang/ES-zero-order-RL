#!/usr/bin/env bash
# Countdown on Qwen2.5-7B-Instruct: N=30, B=100, steps=100, eval every 10 steps, rho per arm.
#
#   stage 0  untrained base reference   (countdown ID + the six math sets as OOD)
#   stage 2  ES vanilla                 -\  SERIAL, each arm alone on BOTH GPUs (TP=2)
#   stage 3  ES baseaxis                -/
#   stage 4  readout: accuracy + wall clock + whole-card peak GPU memory per method
#
# Every stage runs to completion before the next starts. No two stages ever share a card, so
# the per-arm memory and throughput numbers are attributable.
#
# WHY 7B
# ------
# PAPER_COUNTDOWN_SETUP.md: the paper's Table 1 reports Qwen-2.5-7B countdown 31.2% base ->
# 66.8% ES. That is the highest base accuracy of any model in the table, so it is the one row
# where "did ES help" is asked of a model that can already do the task -- unlike the 1.5B row
# (0.7% base), where almost any movement looks like a win. B=100 here matches the 08-16 run
# rather than the B=1000 run now on the cards, because at 7B the B=1000 shape is ~10x the
# compute and does not fit the schedule.
#
# MEMORY BUDGET (worked out, not guessed) -- per card, 143771 MiB available
# -------------------------------------------------------------------------
#   vLLM fp16 weights, TP=2 .................  7265 MiB   (15.23 GB / 2)
#   es_worker._es_base, fp32, TP=2 .......... 14530 MiB   <- fp32, NOT fp16; see es_worker.py:19-26
#   axis_worker._theta0, fp32 (baseaxis only) 14530 MiB
#   vLLM budget at util=0.60 ................ 86263 MiB   (weights + KV, inside vLLM's accounting)
#   baseaxis worst case ..................... 86263 + 29060 = 115323 MiB  (80% of the card)
# The two fp32 snapshots live OUTSIDE vLLM's gpu_memory_utilization accounting, which is why
# util cannot be left at the 0.85 the 1.5B drivers use: 0.85 would reserve 122205 MiB and the
# baseaxis snapshots would then push the card past its 143771 MiB. Actual KV demand is small
# (MINI_BATCH=64 x ~2250 tok x 28 KiB/tok/card ~ 4 GB), so 0.60 is simultaneously safe and far
# more KV than the run can use. Both arms use the SAME util so their KV budgets are comparable.
#
# COST -- READ BEFORE LAUNCHING
# -----------------------------
# There is NO measured 7B throughput on this box. The 0.30 s/generation constant below is an
# extrapolation from the 1.5B TP=2 measurement (0.1056 s/gen, from the B=1000 run's 52.8 min/step)
# scaled by ~3x for the parameter count. Treat it as a factor-of-2 estimate. Run SMOKE=1 first:
# it does 2 steps at eval_cap 50 into a separate _smoke tag and prints the real s/step.
#
# Usage:  DRY_RUN=1 ./run_countdown_7b.sh        # plan only, no GPU
#         SMOKE=1   ./run_countdown_7b.sh        # 2 steps, cap 50, calibrates s/step
#         ./run_countdown_7b.sh                  # full run, single seed
#         STAGES="0 2 4" ./run_countdown_7b.sh   # base + vanilla + readout (skip baseaxis)
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

# Every downstream path keys off MODEL via config.RESULTS_SUFFIX (-> _qwen25_7b_instruct), so
# setting it here is what scopes the results dirs. The weights are NOT in this user's HF cache
# (13 GB free on /home, weights are 15.23 GB); ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
# is a symlink to the read-only shared store /models/models--Qwen--Qwen2.5-7B-Instruct. HF_HUB_CACHE
# is deliberately left at its default: /models holds no datasets--* dirs, so redirecting the cache
# root would break every load_dataset() in eval_core.py.
export MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

STAGES="${STAGES:-0 2 3 4}"
GPU="${GPU:-0,1}"
TP="${TP:-2}"
SEEDS="${SEEDS:-0}"
N="${N:-30}"
B="${B:-100}"                 # 08-16's batch, not the B=1000 run's
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"
MAXTOK="${MAXTOK:-2048}"      # 07-30 protocol: SAME cap for training and eval, never mixed
EVAL_CAP="${EVAL_CAP:-300}"
SIGMA="${SIGMA:-1e-3}"; ALPHA="${ALPHA:-5e-4}"    # paper's values, unchanged
MINI_BATCH="${MINI_BATCH:-64}"  # vLLM generate chunk; bounds peak KV
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
MEM_POLL="${MEM_POLL:-10}"      # seconds between nvidia-smi samples
RHO_SPLITS="${RHO_SPLITS:-20}"  # permutations averaged per step by rho_curve.py
S_PER_GEN="${S_PER_GEN:-0.30}"  # projection constant ONLY; see the cost note above

ES_MEM_UTIL="${ES_MEM_UTIL:-0.60}"   # see MEMORY BUDGET above; do not raise without redoing it

TAG="countdown_7b_N${N}_B${B}"
if [[ "$SMOKE" == "1" ]]; then
  STEPS=2; EVAL_CAP=50; EVAL_EVERY=1; TAG="${TAG}_smoke"
fi

PER_STEP=$((N * B))
ES_GENERATIONS=$((STEPS * PER_STEP))

RES="axis_probe/results/${TAG}"
LOGS="axis_probe/logs/${TAG}"
CKPT="${CKPT:-/home/hyin66/es_ckpts/${TAG}}"    # NOT /tmp: the 08-19 run lost its checkpoints
mkdir -p "$RES" "$LOGS" "$CKPT"
DRIVER="$LOGS/driver.log"
MEMCSV="$LOGS/gpu_mem.csv"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

# ---- preflight: fail loudly here rather than 40 minutes into stage 2 ----
preflight() {
  local bad=0
  local snap
  snap=$(HF_HUB_OFFLINE=1 "$P4_PY" - <<'PY' 2>/dev/null
try:
    from huggingface_hub import snapshot_download
    import glob, os
    p = snapshot_download(os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"), local_files_only=True)
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
  # both cards must be essentially free -- the B=1000 run holds ~124 GB/card while it lasts
  IFS=',' read -ra GPUS <<< "$GPU"
  for g in "${GPUS[@]}"; do
    local used
    used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    say "PREFLIGHT gpu$g: ${used:-?} MiB in use"
    if [[ -n "${used:-}" && $used -gt 5000 ]]; then
      say "PREFLIGHT FAIL: gpu$g has ${used} MiB in use; this run needs both cards to itself"; bad=1
    fi
  done
  return $bad
}

# ---- plan ----
plan() {
  local es_h tot ev_h
  es_h=$(awk -v g="$ES_GENERATIONS" -v s="$S_PER_GEN" 'BEGIN{printf "%.1f", g*s/3600}')
  # intermediate evals: one per EVAL_EVERY steps, 7 sets x EVAL_CAP prompts each, plus the final
  ev_h=$(awk -v k="$((STEPS / EVAL_EVERY + 1))" -v c="$EVAL_CAP" -v s="$S_PER_GEN" \
         'BEGIN{printf "%.1f", k*7*c*s/3600}')
  tot=$(awk -v a="$es_h" -v e="$ev_h" -v s="$(echo $SEEDS | wc -w)" 'BEGIN{printf "%.1f", s*2*(a+e)}')
  cat <<EOF | tee -a "$DRIVER"
================ countdown @ Qwen2.5-7B-Instruct ================
model            : $MODEL
stages           : $STAGES        seeds: $SEEDS        smoke: $SMOKE
GPUs             : $GPU (TP=$TP, SERIAL: base -> vanilla -> baseaxis)
ES               : N=$N  B=$B  steps=$STEPS  sigma=$SIGMA  alpha=$ALPHA  mini_batch=$MINI_BATCH
ES generations   : $STEPS x $N x $B = $ES_GENERATIONS   per arm
eval             : every $EVAL_EVERY steps + final, cap=$EVAL_CAP, max_tokens=$MAXTOK (train == eval)
rho              : offline from correct_bits, $RHO_SPLITS splits/step, after each arm
vLLM mem util    : $ES_MEM_UTIL  (fp32 snapshots sit OUTSIDE this; see header)
projection       : train ~${es_h}h + eval ~${ev_h}h per arm  ->  ~${tot}h total
                   ESTIMATE ONLY at ${S_PER_GEN}s/gen -- no measured 7B throughput exists. SMOKE=1 first.
results          : $RES
checkpoints      : $CKPT
=================================================================
EOF
}
plan
if [[ "$DRY_RUN" == "1" ]]; then echo "DRY_RUN=1 -> stopping before any GPU work."; exit 0; fi
preflight || { say "ABORT on preflight. Override with SKIP_PREFLIGHT=1 if you know better."; \
               [[ "${SKIP_PREFLIGHT:-0}" == "1" ]] || exit 3; }

# ---- whole-card memory sampler ----
# MEMORY_REPORT.md (08-13) established the only valid measurement: torch's max_memory_allocated
# is PER PROCESS and misses anything in another pid, so nvidia-smi over the whole card is what
# gets reported. Summed across both TP cards.
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

# rc capture: `echo "rc=$?"` reports the exit code of the LAST expansion in that echo, not of the
# command -- a $(date) on the same line resets it to 0. That bug made every failed GRPO stage
# report rc=0 COMPLETE in the 08-15 run. Capture first, then log.
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

# ---- stage 0: untrained base reference ----
# The anchor every arm is measured against; ES_VS_BASE_HISTORY.md found three of four July trys
# had no such anchor. Run at the SAME TP and util as the arms so the reference and the arms'
# intermediate evals differ only in weights.
if has_stage 0; then
  if [[ -f "$RES/base_summary.json" ]]; then
    say "SKIP stage0 base: $RES/base_summary.json exists"
  else
    run_stage "base" "$LOGS/base.log" \
      "$P4_PY" axis_probe/eval_base.py \
        --out_prefix "$RES/base" --include_countdown \
        --eval_cap "$EVAL_CAP" --max_tokens "$MAXTOK" \
        --gpu "$GPU" --tensor_parallel_size "$TP" --gpu_mem_util "$ES_MEM_UTIL" \
      || say "WARN stage0 failed; later stages still run but lose their reference"
  fi
fi

# ---- stages 2/3: ES vanilla then baseaxis, SERIAL, each on both cards ----
# Same trainer, same data/reward/eval path; only --variant differs. baseaxis additionally holds a
# fp32 theta0 snapshot, which is the ~14.2 GB/card difference the memory readout should show.
es_arm() {
  local variant="$1" seed="$2"
  local pref="$RES/${variant}_N${N}_B${B}_s${seed}"
  if [[ -f "${pref}_summary.json" ]]; then
    say "SKIP ${variant} s${seed}: ${pref}_summary.json exists"
  else
    run_stage "es_${variant}_s${seed}" "$LOGS/${variant}_s${seed}.log" \
      "$P4_PY" axis_probe/src/es_train_axis.py \
        --variant "$variant" --dataset countdown \
        --population_size "$N" --batch "$B" --mini_batch "$MINI_BATCH" \
        --num_steps "$STEPS" --pop_seed "$seed" \
        --sigma "$SIGMA" --alpha "$ALPHA" --max_tokens "$MAXTOK" \
        --gpu "$GPU" --tensor_parallel_size "$TP" \
        --gpu_mem_util "$ES_MEM_UTIL" \
        --eval_cap "$EVAL_CAP" --eval_final --eval_interval "$EVAL_EVERY" --kl \
        --out_prefix "$pref" \
      || say "WARN ${variant} s${seed} failed -- see $LOGS/${variant}_s${seed}.log"
  fi
  # rho reads only the per-step jsonl, so it still produces a curve for a crashed or
  # partially-complete arm (it tolerates a truncated final line).
  if [[ -f "${pref}.jsonl" ]]; then
    say "rho ${variant} s${seed}"
    "$P4_PY" axis_probe/rho_curve.py --jsonl "${pref}.jsonl" --splits "$RHO_SPLITS" \
      > "$LOGS/rho_${variant}_s${seed}.txt" 2>&1
    tail -n 3 "$LOGS/rho_${variant}_s${seed}.txt" | tee -a "$DRIVER"
  else
    say "WARN no ${pref}.jsonl -- no rho curve for ${variant} s${seed}"
  fi
}

for s in $SEEDS; do
  has_stage 2 && es_arm vanilla "$s"
  has_stage 3 && es_arm baseaxis "$s"
done

# ---- stage 4: readout ----
phase "idle"
if has_stage 4; then
  say "START readout"
  # GRPO args zeroed: this line has no GRPO arm. report_countdown_paperB.py is reused rather than
  # forked so the 7B table is laid out identically to the 1.5B one and the two can be read side by side.
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
say "ALL DONE. results=$RES logs=$LOGS  rho=$LOGS/rho_*.txt"
