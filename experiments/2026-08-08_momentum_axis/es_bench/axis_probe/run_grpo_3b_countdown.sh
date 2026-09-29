#!/usr/bin/env bash
# GRPO control for the 3B countdown line, generation-matched to the ES arms, SINGLE GPU.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The 3B countdown ES arms finished 2026-08-30 (COUNTDOWN_3B_VANILLA_VS_BASEAXIS.md):
#   base 0.0800 -> vanilla 0.4700, baseaxis 0.3933, on the pinned countdown[:300] slice.
# There is no GRPO row on that model. 7B GRPO needs two cards (all of bs=4/8/16 OOM on one --
# GRPO_7B_SINGLE_GPU_FAILURE.md), which is why the line moved to 3B in the first place; at 3B the
# FSDP actor is small enough that a single H200 should hold actor + vLLM together. This driver is
# run_grpo_7b_countdown.sh's stage graph with the 2-GPU parameters collapsed to one card and the
# ES shape it is matched against changed to the 3B arms'.
#
# WHAT MATCHES THE 3B ES ARMS (results/countdown_3b_N30_B100/)
# -------------------------------------------------------------
#   model        Qwen/Qwen2.5-3B-Instruct
#   task/data    countdown, grpo/data_countdown (train = rows [300:]; val = the pinned [:300])
#   reward       reward_countdown.py -> the same countdown_task.answer_reward_function the ES
#                arms and eval_core._eval_countdown use
#   prompt       the dataset's raw `context` through the IDENTITY chat template copy
#                (make_raw_template_model.py); the real tokenizer is restored before the math evals
#   max_resp     2048 = the ES arms' max_tokens, and training == eval (07-30 protocol)
#   eval         same battery / cap 300 / extractor / greedy via eval_grpo.py
#   GENERATIONS  100 x 30 x 100 = 300,000, matched exactly (see BUDGET)
#
# WHAT DOES NOT MATCH, AND CANNOT (08-13 budget audit, unchanged)
# ---------------------------------------------------------------
# Matching total generations leaves several axes unequal and ALL of them favour GRPO:
#   * problem draws   ES spends all N x B generations of a step on the SAME B problems (CRN makes
#                     members share the batch); GRPO spends TRAIN_BS x GROUP on TRAIN_BS fresh
#                     ones. Over the run GRPO sees far more distinct problems.
#   * weight updates  586 (at the default bs) against the ES arms' 100 -- 5.9x.
#   * temperature     GRPO rolls out at T=1.0 and evals greedy; ES is greedy throughout.
#   * vLLM KV budget  ES ran at util=0.70, this runs at 0.5 (verl needs the room for the actor),
#                     so the peak-memory columns are NOT a like-for-like method comparison.
# State all of them when citing any GRPO-vs-ES number from this run.
#
# BUDGET
# ------
# ES arm = 100 steps x N=30 x B=100 = 300,000 generations. GRPO at TRAIN_BS x GROUP rollouts/step
# needs 300000/(TRAIN_BS*GROUP) steps to match:
#   TRAIN_BS=64 GROUP=8 -> 512/step ->   586 steps  (default)
#   TRAIN_BS=16 GROUP=8 -> 128/step -> 2,344 steps  (the shape every pre-08-24 GRPO run used)
# BUDGET=manual STEPS=n overrides, and then the run is NOT budget-matched -- the banner says so.
#
# WHY bs=64 IS THE DEFAULT
# ------------------------
# From the 08-24 throughput sweep (7B, 2 cards, GRPO_7B_BS_THROUGHPUT.md): step time is linear in
# rollouts/step with a batch-INDEPENDENT intercept, s/step ~= 11.77 + 0.6790 * rollouts. At a
# FIXED generation budget the intercept is paid `steps` times, so total time falls monotonically
# in bs with no interior optimum. bs also sets the update count, so a larger bs narrows the
# unmatched-updates axis (586 vs 2,344 -- closer to the ES arms' 100).
#   That sweep was 7B on two cards. The SHAPE of the law (fixed intercept, MICRO-bounded
#   activations, sublinear `gen`) is a property of the config, not the model size, but the
#   CONSTANTS here are unmeasured. SMOKE=1 measures them before you commit ~2 days.
# Cost: every prior GRPO result in this repo is bs=8 or 16 and ACTOR_LR=1e-6 was chosen there, so
# bs=64 is a batch/lr regime this repo has not run. Pass TRAIN_BS=16 to reproduce the old shape.
#
# MEMORY (single H200 NVL, 143,771 MiB) -- THE THING THAT KILLED 7B
# ------------------------------------------------------------------
# On one GPU FSDP has nothing to shard across, so the whole actor sits on the card:
#
#   vLLM (weights + KV) at util=0.5 ......... ~71.9 GB   (util x capacity, by construction)
#   FSDP actor, ~12 B/param at 3.09e9 ....... ~37 GB     (bf16 params + bf16 grads + fp32 Adam)
#   activations at MICRO=2, gc on ........... ~10 GB     (logits over a 152k vocab dominate)
#   ------------------------------------------------------------
#   projected peak .......................... ~119 GB    = 83% of the card
#
# The ~12 B/param figure is BACKED OUT of the one 7B config that did not OOM (08-24: TP=2,
# util=0.5, 122.6 GiB/card => ~50.7 GB of actor per card over 3.81e9 sharded params), not
# measured at 3B. The 7B SINGLE-card numbers cannot be used for this: those runs died during
# step 1, before the optimizer states were fully allocated, so their 126-139 GB peaks are lower
# bounds on a footprint that never finished forming.
# Levers if it does not fit, in the order to try them:
#   ROLLOUT_MEM=0.45 -> 0.40   frees ~7 GB per 0.05; KV at 0.40 still holds ~1.3M tokens
#   MICRO=1                    halves the activation term; untested here
#   MAXRESP=1024               breaks train==eval, so only as a last resort
# Conversely, if the smoke peak leaves >25 GB idle, ROLLOUT_MEM=0.6 buys generation throughput.
#
# EVAL: IN-MEMORY CURVE + ONE FINAL-STEP CHECKPOINT
# --------------------------------------------------
# Two eval paths, answering different questions (identical to the 7B driver):
#   1. the curve (no disk).  trainer.test_freq scores the countdown val set against the weights
#      already live in the rollout engine. Stage 3a reads them back out of the log -- zero GPU.
#      An abort therefore still leaves a usable trajectory, which the 08-23 and 08-24 aborts did
#      not. EVAL_EVERY defaults to STEPS/10 so the points land on the SAME fraction-of-budget
#      grid as the ES arms' 10 eval points, which is what makes the two curves overlayable.
#   2. the battery (needs weights on disk).  Six math OOD sets + KL + countdown extract_rate are
#      eval_grpo.py features and require a merged HF model. _validate() only scores val.parquet.
# SAVE_EVERY=last keeps EXACTLY ONE checkpoint, at the last step, so both paths run.
# The two countdown numbers are NOT the same measurement (live weights + val.parquet prompts vs
# merged weights + eval_core prompts; ~2.7 points apart at step 0, measured 08-24 on 7B). Say
# which one you are citing.
#
# extract_rate IS LOAD-BEARING ON THIS TASK. The 3B ES arms' countdown extract_rate fell from
# base 0.8900 to 0.5633 (vanilla) / 0.5500 (baseaxis): non-extracted answers score 0, so every
# countdown accuracy on this line is a LOWER BOUND, and a GRPO-vs-ES gap can come from format
# compliance rather than from solving. Only the battery reports it; the curve cannot.
#
# MODEL IS EXPORTED ON PURPOSE -- DO NOT REMOVE THE `export`
# ----------------------------------------------------------
# eval_grpo.py and kl_capture.py both read config.MODEL, which is os.environ["MODEL"] with a
# fallback to Qwen2.5-Math-1.5B. run_grpo_7b_countdown.sh sets MODEL WITHOUT exporting it, so its
# summaries record model="Qwen/Qwen2.5-Math-1.5B" and its kl_capture.py captured the base record
# from THAT model rather than from the 7B one (eval_grpo.py's own model-agreement guard cannot
# catch it: both sides resolve to the same wrong fallback). preflight_model() below asserts the
# agreement instead of trusting it.
#
# Usage:
#   SMOKE=1 ./run_grpo_3b_countdown.sh                  # 2 steps, cap 40 -- calibrates s/step AND peak memory
#   DRY_RUN=1 ./run_grpo_3b_countdown.sh                # print the plan, touch nothing
#   tmux new -s grpo3b -d './run_grpo_3b_countdown.sh'  # the real run; see the ETA in the banner
#   TRAIN_BS=16 ./run_grpo_3b_countdown.sh              # the pre-08-24 batch shape
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source axis_probe/env.sh

# EXPORTED: config.MODEL reads it. See the MODEL block above.
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

GPUS="${GPUS:-0}"                # single card
NGPU="${NGPU:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"    # no tensor parallelism
TRAIN_BS="${TRAIN_BS:-1024}"       # prompts/step; see WHY bs=64
GROUP="${GROUP:-8}"              # rollouts/prompt = GRPO group size; 8 in every measured config
MICRO="${MICRO:-4}"              # per-GPU micro batch: sets the activation peak, hence the memory
MAXPROMPT="${MAXPROMPT:-1024}"
MAXRESP="${MAXRESP:-2048}"       # = the ES arms' max_tokens; training == eval
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
ROLLOUT_MEM="${ROLLOUT_MEM:-0.5}"
EVAL_GPU="${EVAL_GPU:-${GPUS%%,*}}"   # eval_grpo.py/kl_capture.py take ONE int, so never a list
EVAL_CAP="${EVAL_CAP:-300}"      # = the ES arms' cap; countdown [:300] is the pinned slice
SAVE_EVERY="${SAVE_EVERY:-last}" # last -> ONE ckpt at the final step. 0|none -> curve only.
MEM_POLL="${MEM_POLL:-5}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
BUDGET="${BUDGET:-gen}"
# Projected seconds per rollout, for the ETA only. UNMEASURED AT 3B: interpolated between the
# 1.5B countdown GRPO run (08-15, ~41.5 s/step at 128 rollouts = 0.32 s/rollout, single card) and
# the 7B 2-card sweep. Expect the banner's ETA to be wrong by tens of percent until SMOKE=1 has
# replaced this number -- the 08-23 and 08-24 extrapolations on this line were off by 1.65x.
S_PER_ROLLOUT="${S_PER_ROLLOUT:-0.55}"

# The 3B ES arms' shape, as literals: a config.py edit must not be able to change what this
# control is matched against. Verified against
# countdown_3b_N30_B100/vanilla_N30_B100_s0_summary.json.
ES_N=30; ES_B=100; ES_STEPS=100
ES_GENERATIONS=$((ES_STEPS * ES_N * ES_B))          # 300,000
ES_RES="axis_probe/results/countdown_3b_N30_B100"   # base / vanilla / baseaxis live here

if [[ "$SMOKE" == "1" ]]; then
  BUDGET=manual; STEPS=2; EVAL_CAP=40; EVAL_EVERY=1
fi

ROLLOUTS_PER_STEP=$((TRAIN_BS * GROUP))
case "$BUDGET" in
  # Round UP, not down: 300000/512 = 585.9. Truncating under-resources the control and invites
  # "you starved the baseline"; rounding up overshoots by 0.01%. Both land in the summary.
  gen)    STEPS=$(( (ES_GENERATIONS + ROLLOUTS_PER_STEP - 1) / ROLLOUTS_PER_STEP )) ;;
  manual) STEPS="${STEPS:?BUDGET=manual requires STEPS}" ;;
  *)      echo "unknown BUDGET=$BUDGET (gen|manual)"; exit 2 ;;
esac

# Default the curve onto the ES arms' grid: they evaluated every STEPS/10 updates (10 points at
# 10%..100% of budget), so the same fraction here makes the two curves directly overlayable on a
# generations-consumed x-axis. A hand-set EVAL_EVERY wins.
if [[ -z "${EVAL_EVERY:-}" ]]; then
  EVAL_EVERY=$(( STEPS / 10 ))
  [[ "$EVAL_EVERY" -lt 1 ]] && EVAL_EVERY=1
fi

# SAVE_EVERY is a word OR a number; resolve it once into WANT_CKPT (can the battery run) and
# SAVE_FREQ (what verl is told), so no later test has to re-parse the word -- `[[ last -gt 0 ]]`
# evaluates to false silently rather than erroring, which is the 08-15 rc=$? class of defect.
case "$SAVE_EVERY" in
  last)      WANT_CKPT=1; SAVE_FREQ=$STEPS ;;   # fires once: STEPS % STEPS == 0, and is_last_step
  0|none|-1) WANT_CKPT=0; SAVE_FREQ=-1 ;;
  *[!0-9]*)  echo "!! SAVE_EVERY must be 'last', 'none', or a non-negative integer (got '$SAVE_EVERY')"; exit 2 ;;
  *)         WANT_CKPT=1; SAVE_FREQ=$SAVE_EVERY ;;
esac

GRPO=axis_probe/grpo
SRC=axis_probe/src
DATADIR="$GRPO/data_countdown"                 # model-independent: raw context, no chat template
RAWTMPL="/tmp/rawtmpl_$(basename "$MODEL")"
TAG="grpo_countdown_3b_bs${TRAIN_BS}g${GROUP}_${BUDGET}${STEPS}"
[[ "$SMOKE" == "1" ]] && TAG="${TAG}_smoke"
OUT="axis_probe/results/countdown_3b_grpo"
LOG="axis_probe/logs/countdown_3b_grpo"
# The overlay, not /home/hyin66: a 3B fp32 shard set is ~12 GB and the home fs has been down to
# ~13 GiB free on this box. The overlay is EPHEMERAL (lost on container restart), which is
# tolerable only because stage 3b consumes the checkpoint in the same invocation.
CKPT_ROOT="${CKPT_ROOT:-/var/tmp/es_ckpts}"
CKPT="${CKPT:-$CKPT_ROOT/$TAG}"
MIN_CKPT_GB="${MIN_CKPT_GB:-30}"   # ~12 GB fp32 shards + ~6 GB merged bf16 HF + slack
DRIVER="$LOG/driver.log"
MEMCSV="$LOG/${TAG}_gpu_mem.csv"
STAGE_TIMES="$LOG/${TAG}_stage_times.jsonl"
# Queried, not assumed: the headroom verdict after stage 2 is only meaningful against the real
# capacity, and hardcoding 143771 would silently mis-scale it on any other card.
CARD_MIB="${CARD_MIB:-}"
if [[ -z "$CARD_MIB" ]]; then
  CARD_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPUS%%,*}" 2>/dev/null | head -1)
  [[ "$CARD_MIB" =~ ^[0-9]+$ ]] || CARD_MIB=143771    # H200 NVL, if nvidia-smi is unavailable
fi

TOTAL_GEN=$((STEPS * ROLLOUTS_PER_STEP))

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

# test_freq drives verl's _validate(): >0 evals every n steps AND at the last step; -1 disables.
if [[ "$EVAL_EVERY" -gt 0 ]]; then
  TEST_FREQ=$EVAL_EVERY
  # STEPS/EVAL_EVERY multiples, +1 for the final step when it is not one of them, +1 for the
  # step-0 val_before_train pass. (The 08-24 version of this line forgot step 0 and printed 6
  # where 7 was correct.)
  N_EVALS=$(( 1 + STEPS / EVAL_EVERY + (STEPS % EVAL_EVERY != 0) ))
  EVAL_DESC="in-memory countdown val (300 pinned rows, greedy) at step 0 then every $EVAL_EVERY \
steps + final = $N_EVALS evals (countdown only, no extract_rate)"
  [[ "$WANT_CKPT" == "1" ]] \
    && EVAL_DESC="$EVAL_DESC; PLUS a final merge + six-set battery (+KL, +extract_rate) from the \
step-$STEPS ckpt"
else
  TEST_FREQ=-1
  EVAL_DESC="no in-memory curve (EVAL_EVERY=0)"
  [[ "$WANT_CKPT" == "1" ]] \
    && EVAL_DESC="$EVAL_DESC; only the final merge + six-set battery from the step-$STEPS ckpt" \
    || EVAL_DESC="$EVAL_DESC and no checkpoint either -- this run produces NO accuracy number"
fi

# step 0 baseline: free when we are evaluating anyway, and it is the in-engine counterpart of the
# ES arms' base_summary.json (though not the same measurement -- different prompt assembly).
VAL_BEFORE=$([[ "$EVAL_EVERY" -gt 0 ]] && echo True || echo False)

EST_H=$(awk -v n="$TOTAL_GEN" -v s="$S_PER_ROLLOUT" 'BEGIN{printf "%.1f", n*s/3600}')

BANNER=$(cat <<EOF
================ countdown GRPO @ Qwen2.5-3B-Instruct, SINGLE GPU ================
model            : $MODEL
GPU              : $GPUS (n=$NGPU, rollout TP=$ROLLOUT_TP, no tensor parallelism)
batch            : TRAIN_BS=$TRAIN_BS x GROUP=$GROUP -> $ROLLOUTS_PER_STEP rollouts/step
budget           : BUDGET=$BUDGET STEPS=$STEPS -> $TOTAL_GEN generations (ES arms = $ES_GENERATIONS)
actor            : MICRO=$MICRO lr=$ACTOR_LR kl_coef=$KL_COEF maxprompt=$MAXPROMPT maxresp=$MAXRESP
vLLM mem util    : $ROLLOUT_MEM   (ES arms ran at 0.70 -- peaks are NOT comparable across methods)
eval             : $EVAL_DESC
projection       : ~${EST_H} h at ${S_PER_ROLLOUT}s/rollout -- ESTIMATE, unmeasured at 3B.
                   Run SMOKE=1 first; the two prior extrapolations on this line were off 1.65x.
memory (projected): ~119 GB of $((CARD_MIB / 1024)) GiB (~83%). Levers: ROLLOUT_MEM 0.45/0.40, then MICRO=1.
results          : $OUT/${TAG}_*
compare against  : $ES_RES/  (base 0.0800, vanilla 0.4700, baseaxis 0.3933 on countdown[:300])
==================================================================================
EOF
)

if [[ $TOTAL_GEN -ne $ES_GENERATIONS ]]; then
  D=$((TOTAL_GEN - ES_GENERATIONS))
  PCT=$(awk -v d=$D -v e=$ES_GENERATIONS 'BEGIN{printf "%+.3f%%", 100*d/e}')
  if [[ "$BUDGET" == "gen" ]]; then
    BANNER="$BANNER
NOTE: total_generations differs by $D ($TOTAL_GEN vs $ES_GENERATIONS, $PCT) -- \
$ROLLOUTS_PER_STEP rollouts/step does not divide $ES_GENERATIONS evenly; rounded up."
  else
    BANNER="$BANNER
!! NOT generation-matched: $TOTAL_GEN vs the ES arms' $ES_GENERATIONS ($PCT).
   BUDGET=manual was used. Any accuracy from this run is NOT a budget-matched
   comparison and must be labelled as such."
  fi
fi
echo "$BANNER"

# DRY_RUN exits BEFORE mkdir so it really does touch nothing -- otherwise planning a run would
# create $OUT/$LOG and leave a driver.log that looks like an aborted attempt.
[[ "$DRY_RUN" == "1" ]] && { echo "DRY_RUN=1, stopping before any GPU work."; exit 0; }
mkdir -p "$OUT" "$LOG"
echo "$BANNER" >> "$DRIVER"

# ---- preflight ----
# config.MODEL must agree with $MODEL, or eval_grpo.py records the wrong model name AND
# kl_capture.py captures the base record from the wrong weights. See the MODEL block in the
# header: this is not hypothetical, it is what the 7B GRPO summaries on disk did.
preflight_model() {
  local cfg snap
  cfg=$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(C.MODEL)" 2>/dev/null)
  if [[ "$cfg" != "$MODEL" ]]; then
    say "PREFLIGHT FAIL: config.MODEL='$cfg' but MODEL='$MODEL'."
    say "  kl_capture.py would record the base from '$cfg' and the summary would be mislabelled."
    return 1
  fi
  say "PREFLIGHT config.MODEL == $cfg"
  snap=$(HF_HUB_OFFLINE=1 $P4_PY - <<'PY' 2>/dev/null
try:
    from huggingface_hub import snapshot_download
    import glob, os
    p = snapshot_download(os.environ["MODEL"], local_files_only=True)
    gb = sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(os.path.join(p, "*.safetensors")))/2**30
    print(f"{p} ({gb:.2f} GB)")
except Exception as e:
    print(f"ERR|{e}")
PY
)
  if [[ "$snap" == ERR* || -z "$snap" ]]; then
    say "PREFLIGHT FAIL: $MODEL does not resolve offline ($snap)"; return 1
  fi
  say "PREFLIGHT weights: $snap"
  return 0
}

preflight_gpu() {
  local g used
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>&1)
    # nvidia-smi -i <invalid> returns "No devices were found" -- a multi-word string that trips
    # the numeric test below under `set -u` if it is not caught here first.
    if [[ "$used" =~ ^[0-9]+$ ]]; then
      if [[ "$used" -gt 5000 ]]; then
        say "PREFLIGHT FAIL: GPU $g already holds ${used} MiB. This run needs ~83% of the card,"
        say "  and a shared card makes the sampled peak unattributable."; return 1
      fi
      say "PREFLIGHT gpu$g: ${used} MiB in use"
    else
      say "PREFLIGHT FAIL: GPU $g query failed: $used"; return 1
    fi
  done
  return 0
}

# Checked BEFORE training, because the save happens at the LAST step: an ENOSPC there costs the
# whole run. (The 08-24 gen2344 run was headed for exactly that, on the home fs, undetected.)
preflight_disk() {
  [[ "$WANT_CKPT" == "1" ]] || return 0
  local avail
  mkdir -p "$CKPT"
  avail=$(df -BG --output=avail "$CKPT" 2>/dev/null | tail -1 | tr -dc 0-9)
  if [[ -z "$avail" || "$avail" -lt "$MIN_CKPT_GB" ]]; then
    say "PREFLIGHT FAIL: $CKPT has ${avail:-?} GiB free, need >= $MIN_CKPT_GB"
    say "  (~12 GB fp32 shards + ~6 GB merged HF + slack). Point CKPT_ROOT elsewhere, or"
    say "  SAVE_EVERY=none to run curve-only (drops the six OOD sets, KL and extract_rate)."
    return 1
  fi
  say "PREFLIGHT disk: ${avail} GiB free at $CKPT (need $MIN_CKPT_GB)"
  return 0
}

PF=0
preflight_model || PF=1
preflight_gpu   || PF=1
preflight_disk  || PF=1
if [[ $PF -ne 0 ]]; then
  if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
    say "SKIP_PREFLIGHT=1 -- continuing over the failures above."
  else
    say "ABORT on preflight. Override with SKIP_PREFLIGHT=1 if you know better."; exit 3
  fi
fi

say "start $TAG  steps=$STEPS total_gen=$TOTAL_GEN"
: > "$STAGE_TIMES"

# ---- stage 1: parquet (raw context; model- and extractor-independent, so existence suffices) ----
if [[ -f "$DATADIR/train.parquet" ]]; then
  say "STAGE1 skip (parquet exists: $DATADIR)"
else
  say "STAGE1 build countdown parquet -> $DATADIR"
  $P4_PY "$GRPO/make_countdown_parquet.py" "$DATADIR" 2>&1 | tee -a "$DRIVER"
fi

# ---- stage 2: verl GRPO on one GPU ----
if [[ "$WANT_CKPT" == "1" && -d "$CKPT/global_step_$STEPS/actor" ]]; then
  say "STAGE2 skip (checkpoint for step $STEPS exists)"
  TRAIN_S=0
else
  [[ -f "$RAWTMPL/tokenizer_config.orig.json" ]] \
    || $P4_PY "$GRPO/make_raw_template_model.py" "$MODEL" "$RAWTMPL" 2>&1 | tee -a "$DRIVER"

  # Whole-card sampling, not torch's counter: verl runs vLLM in its own process, so the
  # per-process counter cannot see the KV cache (MEMORY_REPORT.md, 08-13). The peak lands in
  # update_actor and lasts seconds, so a one-off nvidia-smi will usually catch the trough.
  echo "phase,timestamp_epoch,gpu,mem_used_mib" > "$MEMCSV"
  ( while :; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" \
        | while IFS=', ' read -r idx m; do echo "grpo,$ts,$idx,$m" >> "$MEMCSV"; done
      sleep "$MEM_POLL"
    done ) &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null' EXIT

  say "STAGE2 train ($STEPS steps, model=$RAWTMPL)"
  T0=$SECONDS
  CUDA_VISIBLE_DEVICES=$GPUS $P4_PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATADIR/train.parquet" \
    data.val_files="$DATADIR/val.parquet" \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=$MAXPROMPT data.max_response_length=$MAXRESP \
    data.reward_fn_key=data_source data.filter_overlong_prompts=True data.truncation=right \
    actor_rollout_ref.model.path="$RAWTMPL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$GROUP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
    custom_reward_function.path="$GRPO/reward_countdown.py" \
    custom_reward_function.name=compute_score \
    trainer.logger=[console] trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 \
    actor_rollout_ref.actor.checkpoint.save_contents=[model] \
    trainer.default_local_dir="$CKPT" \
    trainer.save_freq=$SAVE_FREQ trainer.test_freq=$TEST_FREQ \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=$VAL_BEFORE \
    trainer.total_epochs=1000 trainer.total_training_steps=$STEPS \
    > "$LOG/${TAG}_train.log" 2>&1
  RC=$?          # capture BEFORE anything else: $(date) in an echo resets $? (08-15 defect)
  TRAIN_S=$((SECONDS - T0))
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; trap - EXIT
  echo "{\"stage\":\"grpo\",\"rc\":$RC,\"seconds\":$TRAIN_S}" >> "$STAGE_TIMES"
  say "STAGE2 rc=$RC elapsed=${TRAIN_S}s ($(awk -v s=$TRAIN_S -v n=$STEPS 'BEGIN{printf "%.1f", s/n}') s/step, \
$(awk -v s=$TRAIN_S -v n=$TOTAL_GEN 'BEGIN{printf "%.3f", s/n}') s/rollout -- feed this back as S_PER_ROLLOUT)"

  $P4_PY - "$MEMCSV" "$CARD_MIB" <<'PY' 2>&1 | tee -a "$DRIVER"
import csv, statistics as st, sys
vals = []
for r in csv.DictReader(open(sys.argv[1])):
    try: vals.append(int(r["mem_used_mib"]))
    except (ValueError, KeyError): pass
if not vals:
    print("  no memory samples"); raise SystemExit
cap = int(sys.argv[2]); pk = max(vals); s = sorted(vals)
print(f"  peak={pk/1024:.2f} GiB  median={st.median(vals)/1024:.2f}  "
      f"p5={s[len(s)//20]/1024:.2f}  n={len(vals)}")
print(f"  headroom={(cap-pk)/1024:.1f} GiB of {cap/1024:.1f} ({100*pk/cap:.1f}% used)")
# The verdict is what makes a smoke run actionable. 2 steps is nearly enough for memory (the
# 08-24 sweep needed 3 before max_memory_reserved plateaued, so read a 2-step peak as a slight
# UNDER-estimate) but nowhere near enough for throughput.
if pk > 0.93 * cap:
    print("  !! >93% of the card. Lower ROLLOUT_MEM by 0.05 before committing to a long run.")
elif pk > 0.88 * cap:
    print("  ~ tight but workable. A longer run may drift up; watch the first hour.")
else:
    print("  ok. Room to raise ROLLOUT_MEM if generation turns out to be the bottleneck.")
print("  (median << peak is expected: the peak is update_actor and lasts seconds)")
PY

  if [[ $RC -ne 0 ]]; then
    say "  !! training failed, see $LOG/${TAG}_train.log"
    grep -m3 -iE "out of memory|CUDA error|Traceback" "$LOG/${TAG}_train.log" | sed 's/^/     /' | tee -a "$DRIVER"
    if grep -qi "out of memory" "$LOG/${TAG}_train.log"; then
      say "  OOM on a single card. Try in this order: ROLLOUT_MEM=0.45, then 0.40, then MICRO=1."
      say "  If none of them fit, 3B GRPO needs two cards -- which is where 7B ended up (08-27)."
    fi
    exit $RC
  fi
fi

# ---- stage 3a: read back the in-memory evals verl already ran (no GPU, no merge) ----
# Written to ${TAG}_curve_* so it cannot collide with stage 3b's ${TAG}_summary.json. The two
# carry DIFFERENT countdown numbers by construction, and overwriting one with the other would
# silently pick a winner. Keep both; cite which one you mean.
if [[ "$EVAL_EVERY" -gt 0 ]]; then
  say "STAGE3a collect in-memory val curve"
  $P4_PY "$GRPO/collect_val_curve.py" "$LOG/${TAG}_train.log" "$OUT/${TAG}_curve" \
      model="$MODEL" task=countdown train_seconds="$TRAIN_S" train_steps="$STEPS" \
      rollouts_per_step="$ROLLOUTS_PER_STEP" total_generations="$TOTAL_GEN" \
      train_bs="$TRAIN_BS" group="$GROUP" eval_every="$EVAL_EVERY" \
      max_response_length="$MAXRESP" seed=0 2>&1 | tee -a "$DRIVER"
  RC3A=${PIPESTATUS[0]}
  say "STAGE3a rc=$RC3A"
  # Non-fatal when the battery is still to come: a parse failure loses the curve, not the run.
  if [[ $RC3A -ne 0 ]]; then
    say "  !! could not parse val metrics from $LOG/${TAG}_train.log"
    [[ "$WANT_CKPT" == "1" ]] || exit $RC3A
  fi
fi

if [[ "$WANT_CKPT" != "1" ]]; then
  if [[ "$EVAL_EVERY" -le 0 ]]; then
    say "STAGE3 skip: EVAL_EVERY=0 and no checkpoint -- no accuracy was measured"
    exit 0
  fi
  say "COMPLETE -> $OUT/${TAG}_curve_summary.json"
  say "  countdown ONLY (in-memory eval; no ckpt -> no OOD, no KL, no extract_rate)."
  exit 0
fi

# ---- stage 3b: merge the one final ckpt + the full ES battery (single GPU; eval_grpo.py pins TP=1) ----
if [[ ! -d "$CKPT/global_step_$STEPS/actor" ]]; then
  say "STAGE3b !! no $CKPT/global_step_$STEPS/actor -- expected one final-step checkpoint"
  say "   (SAVE_EVERY=$SAVE_EVERY). Training may have stopped short of step $STEPS, or the save"
  say "   failed (check for ENOSPC in $LOG/${TAG}_train.log)."
  ls -d "$CKPT"/global_step_* 2>/dev/null | sed 's/^/     found: /' | tee -a "$DRIVER"
  exit 2
fi

# The KL record must come from the UNTRAINED base, in its own process: eval_grpo.py only has the
# merged trained model loaded, so capturing there measures that model against itself (the 08-13
# kl_proxy_drift=-2.11e-7 artifact). Cached per model under results/, reusable by every arm --
# and correct only because MODEL is exported, so kl_capture.py's config.MODEL is the 3B model.
KLREC="$OUT/kl_base_rec.json"
if [[ ! -s "$KLREC" ]]; then
  say "STAGE3b-pre capture base KL record (gpu $EVAL_GPU, model $MODEL)"
  $P4_PY "$SRC/kl_capture.py" "$KLREC" --gpu "$EVAL_GPU" --max_tokens "$MAXRESP" \
      > "$LOG/${TAG}_klcapture.log" 2>&1
  RCK=$?
  say "STAGE3b-pre rc=$RCK"
  if [[ $RCK -ne 0 ]]; then
    say "  !! base KL capture failed, see $LOG/${TAG}_klcapture.log"
    rm -f "$KLREC"
    exit $RCK
  fi
  grep "self-drift floor" "$LOG/${TAG}_klcapture.log" | sed 's/^/     /' | tee -a "$DRIVER"
else
  say "STAGE3b-pre reuse $KLREC"
fi

say "STAGE3b merge + six-set battery (gpu $EVAL_GPU)"
T3=$SECONDS
$P4_PY "$GRPO/eval_grpo.py" \
    --actor_dir "$CKPT/global_step_$STEPS/actor" \
    --out_prefix "$OUT/$TAG" --eval_cap "$EVAL_CAP" --max_tokens "$MAXRESP" \
    --gpu "$EVAL_GPU" --kl --kl_base_rec "$KLREC" \
    --include_countdown --restore_tokenizer "$RAWTMPL/tokenizer_config.orig.json" \
    --train_seconds "$TRAIN_S" --train_steps "$STEPS" \
    --rollouts_per_step "$ROLLOUTS_PER_STEP" \
    > "$LOG/${TAG}_eval.log" 2>&1
RC3=$?          # capture BEFORE the $(date) inside say() (08-15 defect)
echo "{\"stage\":\"grpo_eval\",\"rc\":$RC3,\"seconds\":$((SECONDS - T3))}" >> "$STAGE_TIMES"
say "STAGE3b rc=$RC3"
if [[ $RC3 -ne 0 ]]; then
  say "  !! battery failed -- NO ${TAG}_summary.json written. See $LOG/${TAG}_eval.log"
  [[ "$EVAL_EVERY" -gt 0 && ${RC3A:-1} -eq 0 ]] \
    && say "     the in-memory curve DID land: $OUT/${TAG}_curve_summary.json"
  exit $RC3
fi
say "COMPLETE -> $OUT/${TAG}_summary.json"

# ---- stage 4: one table with base / GRPO / vanilla / baseaxis ----
# report_countdown_paperB.py keys on filenames (`base_summary.json`, `grpo_summary.json`,
# `*_s*_summary.json`), so a symlink view assembles the four arms WITHOUT writing anything into
# the ES results directory -- that directory is the frozen artifact behind
# COUNTDOWN_3B_VANILLA_VS_BASEAXIS.md and this run must not add files to it.
if [[ "$SMOKE" == "1" ]]; then
  say "STAGE4 skip (SMOKE=1: cap=$EVAL_CAP over $STEPS steps is not comparable to the ES arms)"
elif [[ ! -s "$ES_RES/base_summary.json" ]]; then
  say "STAGE4 skip: no ES arms at $ES_RES -- nothing to tabulate against"
else
  VIEW="$OUT/${TAG}_combined"
  rm -rf "$VIEW"; mkdir -p "$VIEW"
  ln -sf "$(realpath "$OUT/${TAG}_summary.json")" "$VIEW/grpo_summary.json"
  for f in "$ES_RES"/base_summary.json "$ES_RES"/vanilla_*_summary.json "$ES_RES"/baseaxis_*_summary.json; do
    [[ -s "$f" ]] && ln -sf "$(realpath "$f")" "$VIEW/$(basename "$f")"
  done
  say "STAGE4 readout -> $OUT/${TAG}_REPORT.md ($(ls "$VIEW" | wc -l) arms)"
  # The reporter writes --out itself and echoes to stdout; do NOT redirect stdout onto the file.
  $P4_PY axis_probe/report_countdown_paperB.py \
      --results "$VIEW" --logs "$LOG" --mem_csv "$MEMCSV" \
      --stage_times "$STAGE_TIMES" \
      --es_generations "$ES_GENERATIONS" --grpo_rollouts "$TOTAL_GEN" \
      --n "$ES_N" --b "$ES_B" --steps "$ES_STEPS" \
      --grpo_steps "$STEPS" --grpo_rollouts_per_step "$ROLLOUTS_PER_STEP" \
      --es_mem_util 0.70 --grpo_mem_util "$ROLLOUT_MEM" \
      --out "$OUT/${TAG}_REPORT.md" | tee -a "$DRIVER"
  say "  the ES rows' peak-memory and s/step cells read '--': this driver's sampler only covered"
  say "  the GRPO stage. Those numbers are in COUNTDOWN_3B_VANILLA_VS_BASEAXIS.md (117.7/133.7 GB)."
fi

say "ES arms to compare against: $ES_RES/"
say "  countdown[:300] final -- base 0.0800 | vanilla 0.4700 | baseaxis 0.3933"
say "  extract_rate          -- base 0.8900 | vanilla 0.5633 | baseaxis 0.5500  <- accuracies are LOWER BOUNDS"
say "  per-question jsonl is on disk for both sides, so McNemar against either ES arm is paired:"
say "    $OUT/${TAG}_finaleval__countdown_perq.jsonl"
say "    $ES_RES/vanilla_N30_B100_s0_finaleval__countdown_perq.jsonl"
say "  matched on $TOTAL_GEN generations ONLY. Problem draws, update counts ($STEPS vs $ES_STEPS),"
say "  train-time temperature (1.0 vs greedy) and vLLM util ($ROLLOUT_MEM vs 0.70) are all"
say "  unmatched and all favour GRPO. Single seed on both sides."
[[ "$EVAL_EVERY" -gt 0 && ${RC3A:-1} -eq 0 ]] \
  && say "  + in-memory curve: $OUT/${TAG}_curve_summary.json (countdown only, live weights --" \
  && say "    NOT the same measurement as the battery; ~2.7 pt apart at step 0 on 7B)"
CKPT_GB=$(du -sBG "$CKPT" 2>/dev/null | cut -f1 | tr -dc 0-9)
say "  ckpt: $CKPT (${CKPT_GB:-?} GiB, save_contents=[model] so NOT resumable; on the ephemeral"
say "  overlay unless CKPT_ROOT was overridden. Delete once the summary is in hand.)"
