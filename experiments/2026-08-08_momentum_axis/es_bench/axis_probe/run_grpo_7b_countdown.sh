#!/usr/bin/env bash
# GRPO control for the 7B countdown line, generation-matched to the ES vanilla arm, 2 GPUs.
#
# WHY A SEPARATE SCRIPT INSTEAD OF run_grpo.sh
# --------------------------------------------
# run_grpo.sh hardcodes single-GPU training (`tensor_model_parallel_size=1`,
# `n_gpus_per_node=1`) and is the frozen control for the four MATH-line ES arms plus the 08-16
# 1.5B countdown run. Adding TP/n_gpus knobs there would mean those results were produced by a
# different file than the one on disk. This driver duplicates the stage-2 verl invocation with
# the 2-GPU parameters and calls the SAME eval_grpo.py for stage 3.
#
# WHAT MATCHES THE ES vanilla ARM (results/countdown_7b_N30_B100/vanilla_N30_B100_s0_summary.json)
# ------------------------------------------------------------------------------------------------
#   model        Qwen/Qwen2.5-7B-Instruct
#   task/data    countdown, grpo/data_countdown (rows [300:]; eval pins [:300])
#   reward       reward_countdown.py -> the same countdown_task.answer_reward_function the ES
#                arms and eval_core._eval_countdown use
#   prompt       the dataset's raw `context` via the IDENTITY chat template copy
#                (make_raw_template_model.py), tokenizer restored before the math OOD evals
#   max_resp     2048 = the ES arm's max_tokens, and training == eval (07-30 protocol)
#   eval         same battery/cap(300)/extractor via eval_grpo.py
#   GENERATIONS  30 x 100 x 100 = 300,000, matched exactly (see BUDGET below)
#
# WHAT DOES NOT MATCH, AND CANNOT
# -------------------------------
# The 08-13 budget audit applies unchanged: matching total generations leaves TWO axes unequal
# and BOTH favour GRPO.
#   * problem draws   ES spends all N x B generations of a step on the SAME B problems (CRN
#                     requires members share the batch); GRPO spends TRAIN_BS x GROUP on
#                     TRAIN_BS fresh problems. Over the run GRPO sees ~2x the distinct problems.
#   * weight updates  one per step for both, but GRPO gets 586 steps to ES's 100 (5.9x). At the
#                     pre-08-24 default TRAIN_BS=16 this was 2,343 steps = 23x, so the larger
#                     batch narrows this particular unmatched axis as a side effect.
# Also unmatched: GRPO trains at T=1.0 and evals greedy; ES is greedy throughout. State all
# three when citing any GRPO-vs-ES number from this run.
#
# BUDGET
# ------
# ES arm = 100 steps x N=30 x B=100 = 300,000 generations.
# GRPO at TRAIN_BS x GROUP rollouts/step needs 300000/(TRAIN_BS*GROUP) steps to match:
#   TRAIN_BS=64 GROUP=8 -> 512/step ->   586 steps  (default; fastest measured, see COST)
#   TRAIN_BS=16 GROUP=8 -> 128/step -> 2,343 steps  (pre-08-24 default; 63.5 h)
# BUDGET=manual STEPS=n overrides for smoke tests.
#
# EVAL: IN-MEMORY CURVE + ONE FINAL-STEP CHECKPOINT (changed 2026-08-24)
# ----------------------------------------------------------------------
# TWO eval paths run, and they answer different questions:
#
#   1. the curve (no disk).  `trainer.test_freq=$EVAL_EVERY` makes verl's own `_validate()`
#      score the countdown val set against the weights already live in the rollout engine
#      (trainer->rollout is pushed every step via `update_weights`, pure IPC). No FSDP shards,
#      no HF merge. Stage 3 reads those numbers back out of the log -- zero GPU, seconds.
#        EVAL_EVERY=n  -> every n steps AND at the last step (verl's `is_last_step` clause)
#        EVAL_EVERY=0  -> test_freq=-1, no curve at all (memory/throughput probes)
#      An abort therefore still leaves a usable trajectory, which is what the 08-23 (B=1000)
#      and 08-24 (baseaxis) aborts did NOT leave.
#
#   2. the battery (needs weights on disk).  The six math OOD sets, the KL proxy and countdown
#      `extract_rate` are eval_grpo.py features and require a merged HF model. `_validate()`
#      only ever scores val.parquet, so the curve cannot produce them.
#
# SAVE_EVERY=last (default) keeps EXACTLY ONE checkpoint, at the last step, so both paths run:
#   verl saves when `save_freq > 0 && (is_last_step || step % save_freq == 0)`, so save_freq
#   = $STEPS fires once, at step $STEPS. `max_actor_ckpt_to_keep=1` is belt-and-braces.
#   SAVE_EVERY=0|none -> save_freq=-1, curve only (and NO resume point, NO OOD columns).
#   SAVE_EVERY=n      -> every n steps, the pre-08-24 behaviour. Costs n-fold disk.
#
# DISK: this is the binding constraint, and the earlier "13 GiB per checkpoint" note was wrong.
#   FSDP SHARDED_STATE_DICT stores fp32 params, so a 7B model shard set is ~28 GB, and Adam
#   states would add ~56 GB more. Hence:
#     * `checkpoint.save_contents=[model]` -- optimizer/extra dropped. verl.model_merger only
#       globs `model_world_size_*_rank_*.pt`, so the merge and the battery are unaffected. The
#       cost is that the checkpoint is NOT resumable; it exists to be evaluated, not restarted.
#     * CKPT_ROOT defaults to the overlay (~172 GiB free), NOT /home/hyin66 (13 GiB free -- one
#       checkpoint does not fit, and ENOSPC mid-save is exactly how the 08-24 run was going to
#       die). The overlay is EPHEMERAL: it does not survive a container restart. That is
#       tolerable only because stage 3 runs immediately after stage 2 in the same invocation.
#       Pass CKPT=<persistent path> if you need it to outlive the run and have the space.
#   Budget ~28 GB shards + ~15 GB merged HF (bf16). MIN_CKPT_GB preflights for it.
#
# val set = `$DATADIR/val.parquet` = countdown rows [:300], the SAME pinned slice
# eval_core._eval_countdown scores, graded by the same reward_countdown.py ->
# countdown_task.answer_reward_function. verl's val defaults are temperature=0, do_sample=False,
# n=1 (greedy), matching the ES arms' eval protocol.
#
# ONE CONFOUND THE CURVE AVOIDS AND THE BATTERY DOES NOT
#   The 08-13 audit could not rule out FSDP->HF merge dtype rounding as the cause of GRPO's ID
#   drop. The curve never merges, so its countdown column is free of that explanation; the
#   battery's columns are not. When the two disagree on countdown, say which path produced which
#   number -- they also differ in prompt assembly (val.parquet vs eval_core's construction),
#   which is worth ~2.7 points at step 0 (measured 08-24).
#
# MEMORY (2 x H200 NVL, 143,771 MiB/card)
# ---------------------------------------
# The 08-24 sweep measured this exact config at GROUP=8, MICRO=2, MAXRESP=2048, util=0.5:
# 122.8-122.9 GiB/card whole-card peak, FLAT across TRAIN_BS 4..64 -- the spread over an 8x
# batch range was 0.08 GiB, inside vLLM's own allocation jitter. bs buys time, not memory:
# peak activation is set by MICRO (2), and a larger bs just runs more micro-batches in sequence.
# So TRAIN_BS=64 has the same ~20 GiB/card headroom as 16 did, and no bs in that range OOMs.
# Peak lands in update_actor and lasts seconds -- a one-off nvidia-smi will usually show the
# ~50 GiB trough instead, so this script samples every MEM_POLL seconds and reports max.
#
# COST -- why TRAIN_BS=64 is the default (measured 08-24, GRPO_7B_BS_THROUGHPUT.md)
# ---------------------------------------------------------------------------------
# Steady-state step time is linear in rollouts/step with a batch-INDEPENDENT intercept:
#
#     s/step ~= 11.77 + 0.6790 * rollouts_per_step        (residuals 1-3% over bs 4..64)
#
# update_actor / old_log_prob / ref scale strictly proportionally (MICRO is fixed, so bs only
# adds sequential micro-batches -- no economy of scale). The ONLY scale win is `gen`, which is
# strongly sublinear: 16x the rollouts costs 3.03x the time, because at small batch vLLM's
# continuous batching cannot saturate the GPU. `reward` is constant (CPU-side) and forms the
# bulk of that 11.77 s intercept.
#
# At a FIXED 300,000 generations the intercept is paid `steps` times, so total time falls
# monotonically in bs -- there is no interior optimum:
#
#     bs   rollouts/step   steps    s/step   TOTAL      intercept share
#      4        32         9,375      33.8    88.1 h        35%
#      8        64         4,688      54.6    71.1 h        22%
#     16       128         2,344      97.6    63.5 h        12%
#     32       256         1,172     187.7    61.1 h         6%
#     64       512           586     358.7    58.4 h         3%   <- default
#
# Zeroing the intercept gives an asymptotic floor of 0.6790 * 300000 = 56.6 h, so bs=64 is
# already within 3.2% of it; bs=128 would save a further 0.9 h (untested, extrapolated) while
# cutting updates to 293. bs=64 saves 5.1 h (8%) over the old bs=16 default.
#
# TRAIN_BS IS NOT A PURE SPEED KNOB. At fixed generations it also sets the number of weight
# updates (586 at bs=64 vs 2,344 at bs=16). That cuts the unmatched-updates axis from 23x to
# 5.9x versus the ES arm's 100 -- better for budget fairness -- but it also moves GRPO into a
# batch/lr regime this repo has never run (every prior GRPO result is bs=8 or 16, and
# ACTOR_LR=1e-6 was chosen there). Any accuracy difference against earlier GRPO rows is
# confounded by that. Pass TRAIN_BS=32 (61.1 h) for a middle ground, or TRAIN_BS=16 to
# reproduce the old configuration exactly.
#
# The only long-run cross-check of the table: the killed gen2344 run (bs=16) measured
# 100.0 s/step against the table's 97.6 -- 2.5% apart. Each row is a single 5-step run with no
# variance estimate, and response lengths drift over a real run, so treat 58.4 h as an estimate
# and use the rate this run prints every 50 steps.
#
# Usage:
#   tmux new -s grpo7b -d './run_grpo_7b_countdown.sh 2>&1 | tee logs/...'   # see below
#   DRY_RUN=1 ./run_grpo_7b_countdown.sh          # print the plan, touch nothing
#   BUDGET=manual STEPS=20 ./run_grpo_7b_countdown.sh   # smoke
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source axis_probe/env.sh

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPUS="${GPUS:-0}"
NGPU="${NGPU:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
TRAIN_BS="${TRAIN_BS:-8}"       # 08-24 throughput sweep: fastest at fixed generations. See COST.
GROUP="${GROUP:-16}"
MICRO="${MICRO:-1}"              # 8 OOMs on 7B; 2 is what the 08-24 sweep validated
MAXPROMPT="${MAXPROMPT:-1024}"
MAXRESP="${MAXRESP:-2048}"       # = ES arm max_tokens; training == eval
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
ROLLOUT_MEM="${ROLLOUT_MEM:-0.5}"
EVAL_EVERY="${EVAL_EVERY:-100}"  # test_freq: eval every n steps + at the last step. 0 disables.
                                 # At bs=64 -> 586 steps this is 7 evals (0,100..500,586).
EVAL_GPU="${EVAL_GPU:-0}"        # used by the final on-disk battery (SAVE_EVERY=last or >0)
EVAL_CAP="${EVAL_CAP:-300}"      # = the ES arms' cap; countdown [:300] is the pinned slice
# last (default) = ONE checkpoint at the final step -> curve + full six-set battery.
# 0|none         = no checkpoint at all -> curve only, no OOD/KL/extract_rate, no resume point.
# n              = every n steps (pre-08-24 behaviour), n-fold disk.
SAVE_EVERY="${SAVE_EVERY:-last}"
MEM_POLL="${MEM_POLL:-5}"
DRY_RUN="${DRY_RUN:-0}"
BUDGET="${BUDGET:-gen}"

# The ES vanilla arm's shape, stated as literals so a config.py edit cannot silently change
# what this control is matched against. Verified against the arm's summary.json.
ES_N=30; ES_B=100; ES_STEPS=100
ES_GENERATIONS=$((ES_STEPS * ES_N * ES_B))          # 300,000
ROLLOUTS_PER_STEP=$((TRAIN_BS * GROUP))
case "$BUDGET" in
  # Round UP, not down: 300000/128 = 2343.75. Truncating gives 299,904 (96 short) and invites
  # "you under-resourced the baseline"; rounding up gives 300,032 (32 over, 0.01%). Erring
  # toward the control is the defensible direction. Both are stated in the summary.
  gen)    STEPS=$(( (ES_GENERATIONS + ROLLOUTS_PER_STEP - 1) / ROLLOUTS_PER_STEP )) ;;
  manual) STEPS="${STEPS:?BUDGET=manual requires STEPS}" ;;
  *)      echo "unknown BUDGET=$BUDGET (gen|manual)"; exit 2 ;;
esac

# SAVE_EVERY is a word OR a number; resolve it into the two things the rest of the script needs:
# WANT_CKPT (does a step-$STEPS checkpoint exist afterwards -> can the battery run) and SAVE_FREQ
# (what verl is told). Done here, once, so no later test has to re-parse the word -- `[[ last -gt
# 0 ]]` silently evaluates to false rather than erroring, which is exactly the kind of quiet
# wrong-branch the 08-15 rc=$? defect was.
case "$SAVE_EVERY" in
  last)      WANT_CKPT=1; SAVE_FREQ=$STEPS ;;    # fires once: STEPS % STEPS == 0, and is_last_step
  0|none|-1) WANT_CKPT=0; SAVE_FREQ=-1 ;;
  *[!0-9]*)  echo "!! SAVE_EVERY must be 'last', 'none', or a non-negative integer (got '$SAVE_EVERY')"; exit 2 ;;
  *)         WANT_CKPT=1; SAVE_FREQ=$SAVE_EVERY ;;
esac

GRPO=axis_probe/grpo
SRC=axis_probe/src
DATADIR="$GRPO/data_countdown"
RAWTMPL="/tmp/rawtmpl_$(basename "$MODEL")"
TAG="grpo_countdown_7b_bs${TRAIN_BS}g${GROUP}_${BUDGET}${STEPS}"
OUT="axis_probe/results/countdown_7b_grpo"
LOG="axis_probe/logs/countdown_7b_grpo"
# Default root is the OVERLAY, not /home/hyin66: the home fs has ~13 GiB free and one 7B fp32
# model shard set is ~28 GB, so a save there ENOSPCs mid-write. The overlay is ephemeral (lost on
# container restart) and that is acceptable ONLY because stage 3 consumes the checkpoint in the
# same invocation. Override CKPT to keep it longer -- and check `df` first.
CKPT_ROOT="${CKPT_ROOT:-/var/tmp/es_ckpts}"
CKPT="${CKPT:-$CKPT_ROOT/$TAG}"
MIN_CKPT_GB="${MIN_CKPT_GB:-60}"   # ~28 GB fp32 shards + ~15 GB merged bf16 HF + slack
DRIVER="$LOG/driver.log"
MEMCSV="$LOG/${TAG}_gpu_mem.csv"

TOTAL_GEN=$((STEPS * ROLLOUTS_PER_STEP))

# test_freq drives verl's _validate(): >0 evals every n steps AND at the last step; -1 disables.
if [[ "$EVAL_EVERY" -gt 0 ]]; then
  TEST_FREQ=$EVAL_EVERY
  # STEPS/EVAL_EVERY multiples, +1 for the final step when it is not one of them, +1 for the
  # step-0 val_before_train pass (which this branch always enables -- see VAL_BEFORE below).
  N_EVALS=$(( 1 + STEPS / EVAL_EVERY + (STEPS % EVAL_EVERY != 0) ))
  EVAL_DESC="in-memory countdown val (300 pinned rows, greedy) at step 0 then every $EVAL_EVERY \
steps + final = $N_EVALS evals (countdown only)"
  [[ "$WANT_CKPT" == "1" ]] \
    && EVAL_DESC="$EVAL_DESC; PLUS a final merge + full six-set battery (+KL, +extract_rate) \
from the step-$STEPS ckpt"
else
  TEST_FREQ=-1
  EVAL_DESC="no in-memory curve (EVAL_EVERY=0)"
  [[ "$WANT_CKPT" == "1" ]] \
    && EVAL_DESC="$EVAL_DESC; only the final merge + six-set battery from the step-$STEPS ckpt" \
    || EVAL_DESC="$EVAL_DESC and no checkpoint either -- this run produces NO accuracy number"
fi

# step 0 baseline: free when we are evaluating anyway, and it is the untrained reference the
# ES arms get from their base_summary.json. Skip it when eval is off.
VAL_BEFORE=$([[ "$EVAL_EVERY" -gt 0 ]] && echo True || echo False)

if [[ "$WANT_CKPT" == "1" ]]; then
  AVAIL_GB=$(df -BG --output=avail "$(dirname "$CKPT_ROOT")" | tail -1 | tr -dc 0-9)
  CKPT_DESC="save_freq=$SAVE_FREQ ($SAVE_EVERY) -> $CKPT, save_contents=[model] only \
(~28 GB fp32 shards, NOT resumable; +~15 GB merged HF). ${AVAIL_GB} GiB free on $(dirname "$CKPT_ROOT")"
  [[ "$SAVE_EVERY" == "last" ]] && CKPT_DESC="$CKPT_DESC -- exactly ONE ckpt, at step $STEPS"
else
  CKPT_DESC="none (save_freq=-1). Eval reads the live rollout weights, so nothing lands on disk. \
Consequence: NO resume point, and NO math OOD / KL / extract_rate columns."
fi
cat <<EOF
model=$MODEL  task=countdown  gpus=$GPUS (n=$NGPU, rollout TP=$ROLLOUT_TP)
TRAIN_BS=$TRAIN_BS GROUP=$GROUP -> $ROLLOUTS_PER_STEP rollouts/step
BUDGET=$BUDGET STEPS=$STEPS -> total_generations=$TOTAL_GEN  (ES vanilla arm = $ES_GENERATIONS)
MICRO=$MICRO maxprompt=$MAXPROMPT maxresp=$MAXRESP util=$ROLLOUT_MEM
eval: $EVAL_DESC
ckpt: $CKPT_DESC
out=$OUT
EOF
if [[ $TOTAL_GEN -ne $ES_GENERATIONS ]]; then
  D=$((TOTAL_GEN - ES_GENERATIONS))
  PCT=$(awk -v d=$D -v e=$ES_GENERATIONS 'BEGIN{printf "%+.3f%%", 100*d/e}')
  if [[ "$BUDGET" == "gen" ]]; then
    echo "NOTE: total_generations differs by $D ($TOTAL_GEN vs $ES_GENERATIONS, $PCT) --" \
         "$ROLLOUTS_PER_STEP rollouts/step does not divide $ES_GENERATIONS evenly; rounded up."
  else
    echo "!! NOT generation-matched: $TOTAL_GEN vs the ES arm's $ES_GENERATIONS ($PCT)."
    echo "   BUDGET=manual was used. Any accuracy from this run is NOT a budget-matched"
    echo "   comparison and must be labelled as such."
  fi
fi
[[ "$DRY_RUN" == "1" ]] && { echo "DRY_RUN=1, stopping."; exit 0; }

mkdir -p "$OUT" "$LOG"
[[ "$WANT_CKPT" == "1" ]] && mkdir -p "$CKPT"

# preflight: each listed GPU must be idle, else the sampled peak is not attributable to this run
# (same criterion as probe_grpo_7b_bs.sh / gpu_peak.py)
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>&1)
  # nvidia-smi -i <invalid> returns "No devices were found" (multi-word string), which triggers
  # "unbound variable" under `set -u` if not quoted. Catch it explicitly.
  if [[ "$used" =~ ^[0-9]+$ ]]; then
    if [[ "$used" -gt 5000 ]]; then
      echo "!! GPU $g already holds ${used} MiB. Free it first, or the peak is unattributable."; exit 2
    fi
  else
    echo "!! GPU $g query failed: $used"; exit 2
  fi
done

# preflight: disk for the checkpoint. Checked BEFORE training, because the save happens at the
# LAST step -- an ENOSPC there costs the whole run. (The 08-24 gen2344 run was headed for exactly
# that at step 200, on the home fs, undetected.)
if [[ "$WANT_CKPT" == "1" ]]; then
  avail=$(df -BG --output=avail "$CKPT" 2>/dev/null | tail -1 | tr -dc 0-9)
  if [[ -z "$avail" || "$avail" -lt "$MIN_CKPT_GB" ]]; then
    echo "!! $CKPT has ${avail:-?} GiB free, need >= $MIN_CKPT_GB (~28 GB fp32 shards + ~15 GB"
    echo "   merged HF + slack). Point CKPT_ROOT at a bigger filesystem, or SAVE_EVERY=none to"
    echo "   run curve-only (which drops the six OOD sets, KL and extract_rate)."; exit 2
  fi
fi

echo "[$(date +%F_%T)] start $TAG  steps=$STEPS total_gen=$TOTAL_GEN" | tee "$DRIVER"

# ---- stage 1: parquet (raw context; model- and extractor-independent, so existence suffices) ----
if [[ -f "$DATADIR/train.parquet" ]]; then
  echo "[$(date +%F_%T)] STAGE1 skip (parquet exists)" | tee -a "$DRIVER"
else
  echo "[$(date +%F_%T)] STAGE1 build countdown parquet" | tee -a "$DRIVER"
  $P4_PY "$GRPO/make_countdown_parquet.py" "$DATADIR" 2>&1 | tee -a "$DRIVER"
fi

# ---- stage 2: verl GRPO on 2 GPUs ----
if [[ "$WANT_CKPT" == "1" && -d "$CKPT/global_step_$STEPS/actor" ]]; then
  echo "[$(date +%F_%T)] STAGE2 skip (checkpoint for step $STEPS exists)" | tee -a "$DRIVER"
  TRAIN_S=0
else
  [[ -f "$RAWTMPL/tokenizer_config.orig.json" ]] \
    || $P4_PY "$GRPO/make_raw_template_model.py" "$MODEL" "$RAWTMPL" 2>&1 | tee -a "$DRIVER"

  echo "timestamp_epoch,gpu,mem_used_mib" > "$MEMCSV"
  ( while :; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" \
        | while IFS=', ' read -r idx m; do echo "$ts,$idx,$m" >> "$MEMCSV"; done
      sleep "$MEM_POLL"
    done ) &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null' EXIT

  echo "[$(date +%F_%T)] STAGE2 train ($STEPS steps, model=$RAWTMPL)" | tee -a "$DRIVER"
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
  echo "[$(date +%F_%T)] STAGE2 rc=$RC elapsed=${TRAIN_S}s ($(awk -v s=$TRAIN_S -v n=$STEPS 'BEGIN{printf "%.1f", s/n}') s/step)" | tee -a "$DRIVER"

  $P4_PY - "$MEMCSV" <<'PY' 2>&1 | tee -a "$DRIVER"
import csv, collections, statistics as st, sys
per, tot = collections.defaultdict(list), collections.defaultdict(int)
for r in csv.DictReader(open(sys.argv[1])):
    try:
        per[r["gpu"]].append(int(r["mem_used_mib"])); tot[r["timestamp_epoch"]] += int(r["mem_used_mib"])
    except (ValueError, KeyError): pass
for g, v in sorted(per.items()):
    s = sorted(v)
    print(f"  gpu{g} peak={max(v)/1024:.2f} GiB  median={st.median(v)/1024:.2f}  "
          f"p5={s[len(s)//20]/1024:.2f}  n={len(v)}")
if tot: print(f"  both cards peak={max(tot.values())/1024:.2f} GiB")
print("  (median << peak is expected: the peak is update_actor and lasts seconds)")
PY

  if [[ $RC -ne 0 ]]; then
    echo "  !! training failed, see $LOG/${TAG}_train.log" | tee -a "$DRIVER"
    grep -m3 -iE "out of memory|CUDA error|Traceback" "$LOG/${TAG}_train.log" | sed 's/^/     /' | tee -a "$DRIVER"
    exit $RC
  fi
fi

# ---- stage 3a: read back the in-memory evals verl already ran (no GPU, no merge) ----
# Written to ${TAG}_curve_* so it cannot collide with stage 3b's ${TAG}_summary.json. The two
# files carry DIFFERENT countdown numbers by construction (live weights + val.parquet prompts vs
# merged weights + eval_core prompts) and overwriting one with the other would silently pick a
# winner. Keep both; cite which one you mean.
if [[ "$EVAL_EVERY" -gt 0 ]]; then
  echo "[$(date +%F_%T)] STAGE3a collect in-memory val curve" | tee -a "$DRIVER"
  $P4_PY "$GRPO/collect_val_curve.py" "$LOG/${TAG}_train.log" "$OUT/${TAG}_curve" \
      model="$MODEL" task=countdown train_seconds="$TRAIN_S" train_steps="$STEPS" \
      rollouts_per_step="$ROLLOUTS_PER_STEP" total_generations="$TOTAL_GEN" \
      train_bs="$TRAIN_BS" group="$GROUP" eval_every="$EVAL_EVERY" \
      max_response_length="$MAXRESP" seed=0 2>&1 | tee -a "$DRIVER"
  RC3A=${PIPESTATUS[0]}
  echo "[$(date +%F_%T)] STAGE3a rc=$RC3A" | tee -a "$DRIVER"
  # Non-fatal when the battery is still to come: a parse failure loses the curve, not the run.
  if [[ $RC3A -ne 0 ]]; then
    echo "  !! could not parse val metrics from $LOG/${TAG}_train.log" | tee -a "$DRIVER"
    [[ "$WANT_CKPT" == "1" ]] || exit $RC3A
  fi
fi

if [[ "$WANT_CKPT" != "1" ]]; then
  if [[ "$EVAL_EVERY" -le 0 ]]; then
    echo "[$(date +%F_%T)] STAGE3 skip: EVAL_EVERY=0 and no checkpoint, no accuracy was measured" | tee -a "$DRIVER"
    exit 0
  fi
  echo "[$(date +%F_%T)] COMPLETE -> $OUT/${TAG}_curve_summary.json" | tee -a "$DRIVER"
  echo "  countdown ONLY (in-memory eval; no ckpt -> no OOD, no KL, no extract_rate)." | tee -a "$DRIVER"
  echo "  ES vanilla arm to compare against:" | tee -a "$DRIVER"
  echo "    axis_probe/results/countdown_7b_N30_B100/vanilla_N30_B100_s0_summary.json" | tee -a "$DRIVER"
  echo "  both were given $TOTAL_GEN generations; problem draws, update counts and train-time" | tee -a "$DRIVER"
  echo "  temperature are NOT matched, and all three favour GRPO." | tee -a "$DRIVER"
  exit 0
fi

# ---- stage 3b: merge the one final ckpt + the full ES battery (single GPU, eval_grpo.py pins TP=1) ----
if [[ ! -d "$CKPT/global_step_$STEPS/actor" ]]; then
  echo "[$(date +%F_%T)] STAGE3b !! no $CKPT/global_step_$STEPS/actor -- expected one final-step" | tee -a "$DRIVER"
  echo "   checkpoint (SAVE_EVERY=$SAVE_EVERY). Training may have stopped short of step $STEPS," | tee -a "$DRIVER"
  echo "   or the save failed (check ENOSPC in $LOG/${TAG}_train.log)." | tee -a "$DRIVER"
  ls -d "$CKPT"/global_step_* 2>/dev/null | sed 's/^/     found: /' | tee -a "$DRIVER"
  exit 2
fi
# The KL record must come from the UNTRAINED base, in its own process: eval_grpo.py only has the
# merged trained model loaded, and capturing there measures that model against itself (the
# 08-13 kl_proxy_drift=-2.11e-7 artifact). Cached per model under results/, reusable by every arm.
KLREC="$OUT/kl_base_rec.json"
if [[ ! -s "$KLREC" ]]; then
  echo "[$(date +%F_%T)] STAGE3b-pre capture base KL record (gpu $EVAL_GPU)" | tee -a "$DRIVER"
  $P4_PY "$SRC/kl_capture.py" "$KLREC" --gpu "$EVAL_GPU" --max_tokens "$MAXRESP" \
      > "$LOG/${TAG}_klcapture.log" 2>&1
  RCK=$?
  echo "[$(date +%F_%T)] STAGE3b-pre rc=$RCK" | tee -a "$DRIVER"
  if [[ $RCK -ne 0 ]]; then
    echo "  !! base KL capture failed, see $LOG/${TAG}_klcapture.log" | tee -a "$DRIVER"
    rm -f "$KLREC"
    exit $RCK
  fi
  grep "self-drift floor" "$LOG/${TAG}_klcapture.log" | sed 's/^/     /' | tee -a "$DRIVER"
else
  echo "[$(date +%F_%T)] STAGE3b-pre reuse $KLREC" | tee -a "$DRIVER"
fi

echo "[$(date +%F_%T)] STAGE3b merge + six-set battery (gpu $EVAL_GPU)" | tee -a "$DRIVER"
$P4_PY "$GRPO/eval_grpo.py" \
    --actor_dir "$CKPT/global_step_$STEPS/actor" \
    --out_prefix "$OUT/$TAG" --eval_cap "$EVAL_CAP" --max_tokens "$MAXRESP" \
    --gpu "$EVAL_GPU" --kl --kl_base_rec "$KLREC" \
    --include_countdown --restore_tokenizer "$RAWTMPL/tokenizer_config.orig.json" \
    --train_seconds "$TRAIN_S" --train_steps "$STEPS" \
    --rollouts_per_step "$ROLLOUTS_PER_STEP" \
    > "$LOG/${TAG}_eval.log" 2>&1
RC3=$?          # capture BEFORE the $(date) echo below (08-15 defect)
echo "[$(date +%F_%T)] STAGE3b rc=$RC3" | tee -a "$DRIVER"
if [[ $RC3 -ne 0 ]]; then
  echo "  !! battery failed -- NO ${TAG}_summary.json written. See $LOG/${TAG}_eval.log" | tee -a "$DRIVER"
  [[ "$EVAL_EVERY" -gt 0 && ${RC3A:-1} -eq 0 ]] \
    && echo "     the in-memory curve DID land: $OUT/${TAG}_curve_summary.json" | tee -a "$DRIVER"
  exit $RC3
fi
echo "[$(date +%F_%T)] COMPLETE -> $OUT/${TAG}_summary.json" | tee -a "$DRIVER"
[[ "$EVAL_EVERY" -gt 0 && ${RC3A:-1} -eq 0 ]] \
  && echo "  + in-memory curve: $OUT/${TAG}_curve_summary.json (countdown only, live weights)" | tee -a "$DRIVER"
echo "  the two countdown numbers are NOT the same measurement: the curve uses live weights and" | tee -a "$DRIVER"
echo "  val.parquet prompts, the battery uses merged weights and eval_core prompts (~2.7 pt at" | tee -a "$DRIVER"
echo "  step 0, measured 08-24). Say which one you are citing." | tee -a "$DRIVER"
echo "  ES vanilla arm to compare against:" | tee -a "$DRIVER"
echo "    axis_probe/results/countdown_7b_N30_B100/vanilla_N30_B100_s0_summary.json" | tee -a "$DRIVER"
echo "  both were given $TOTAL_GEN generations; problem draws, update counts and train-time" | tee -a "$DRIVER"
echo "  temperature are NOT matched, and all three favour GRPO." | tee -a "$DRIVER"
CKPT_GB=$(du -sBG "$CKPT" 2>/dev/null | cut -f1 | tr -dc 0-9)
echo "  ckpt: $CKPT (${CKPT_GB:-?} GiB, save_contents=[model] so NOT resumable; on the ephemeral" | tee -a "$DRIVER"
echo "  overlay unless you overrode CKPT_ROOT. Delete once the summary is in hand.)" | tee -a "$DRIVER"
