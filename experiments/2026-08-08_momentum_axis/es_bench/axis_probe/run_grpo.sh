#!/usr/bin/env bash
# GRPO control for this try, built to be directly comparable with the momentum / baseaxis /
# vanilla / sigadapt ES arms already in axis_probe/results/.
#
# ---------------------------------------------------------------------------
# WHAT HAS TO MATCH, AND HOW EACH ONE IS ENFORCED
# ---------------------------------------------------------------------------
#   model      config.py MODEL (Qwen2.5-Math-1.5B as of 2026-08-14), fp16   <- shared with ES
#   data       identical problems in identical order       <- stage 1 calls the SAME
#                                                             data_math.make_split(...) the ES
#                                                             trainer calls. NOTE the parquet in
#                                                             2026-07-24/.../grpo_ood/data_math is
#                                                             dated 2026-07-25 and was built with
#                                                             the BROKEN extractor: 0% of its gold
#                                                             answers contain braces, vs 24.7% here.
#                                                             It samples 2000 rows from a 4180-problem
#                                                             pool instead of 5584 and is a DIFFERENT
#                                                             problem set. Do not reuse it.
#   prompt     char-identical rollout string               <- data_math.build_prompt applies the chat
#                                                             template in stage 1; verl gets the
#                                                             finished string as one user turn
#   reward     the same scalar both methods optimise       <- custom_reward_function points at
#                                                             es_bench/shared_reward.py::verl_compute_score
#   response   MAXRESP training tokens (= C.MAX_TOKENS)    <- matching the ES members; eval uses the same cap
#   eval       same battery, cap, extractor, dtype, length <- stage 3 calls eval_core.eval_on_llm
#                                                             exactly as the ES trainer does, and
#                                                             writes the ES `*_summary.json` schema
#
# ---------------------------------------------------------------------------
# BUDGET: the one axis that cannot be matched on all counts at once
# ---------------------------------------------------------------------------
# One ES arm = steps x N x B (math defaults from config.py; was 200x16x8=25,600 @ B=8).
# GRPO produces TRAIN_BS x GROUP rollouts per step, so the two methods can be equated on
# generations OR on wall clock, not both. BUDGET picks which:
#
#   BUDGET=gen    (default) STEPS = (NUM_STEPS*N*B) / (TRAIN_BS*GROUP), B from config.py.
#                 With math defaults (200*16*1024 / 64) that is 51200 steps.
#   BUDGET=steps  STEPS=200, i.e. step-matched. The 2026-07-30 countdown battery used this and
#                 flagged that it UNDERSTATES GRPO, which was ~7x faster per step there.
#   BUDGET=manual STEPS taken from the environment verbatim.
#
# Whichever is chosen, both axes land in the summary (`total_generations`, `wall_clock_s`) so the
# report can state the budget honestly instead of implying a single "fair" comparison exists.
#
#   Usage:  ./run_grpo.sh                          # generation-matched, ~400 steps
#           BUDGET=steps ./run_grpo.sh             # step-matched, 200 steps
#           BUDGET=manual STEPS=50 ./run_grpo.sh   # smoke
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

# Default to config.py rather than a second hardcoded literal: the ES arms this GRPO run is
# the control for read C.MODEL, and a GRPO/ES model mismatch would silently invalidate the
# whole comparison. SUF scopes the results dir by model (see the ES run scripts).
eval "$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(f'CFG_MODEL={C.MODEL}\nSUF={C.RESULTS_SUFFIX}')")"
MODEL="${MODEL:-$CFG_MODEL}"
# TASK=countdown swaps three things and nothing else, so the GRPO definition stays single-sourced:
#   data    countdown.json rows [300:] via make_countdown_parquet.py (eval pins [:300])
#   reward  reward_countdown.py -> the same countdown_task.answer_reward_function the ES arms
#           and eval_core._eval_countdown use
#   prompt  the dataset's raw `context`, which is NOT a chat turn. verl always applies the
#           tokenizer's chat template, so we train against a model copy whose template is the
#           IDENTITY (make_raw_template_model.py) and restore the real tokenizer_config into the
#           merged HF dir afterwards -- the math OOD evals need the true template back.
TASK="${TASK:-math}"
TRAIN_BS="${TRAIN_BS:-8}"          # prompts per step
GROUP="${GROUP:-8}"                # rollouts per prompt (GRPO group size)
# 2048, not 1024: the base model's 4-shot prompt adds ~230 tokens, and 3 of the 2000 training
# prompts then exceed 1024 (longest 1883). verl truncates rather than failing, which would cut
# the question itself and make those rewards meaningless. ES is unaffected (max_model_len 4096).
MAXPROMPT="${MAXPROMPT:-2048}"
MAXRESP="${MAXRESP:-1024}"         # = config.py MAX_TOKENS, the ES training budget
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
MICRO="${MICRO:-4}"              # per-GPU micro batch; verl requires it explicitly
ROLLOUT_MEM="${ROLLOUT_MEM:-0.5}"
GPU="${GPU:-0}"
EVAL_CAP="${EVAL_CAP:-300}"
BUDGET="${BUDGET:-gen}"

# One ES arm = steps x N x B. This default is the MATH line's shape; the countdown line has a
# different one (100 x 30 x 100 = 300,000), so its driver passes ES_GENERATIONS explicitly.
# Without that override BUDGET=gen would silently under-resource GRPO by 12x on countdown.
read _ES_STEPS _ES_N _ES_B <<< "$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(C.NUM_STEPS, 16, C.BATCH_SIZE)")"
ES_GENERATIONS="${ES_GENERATIONS:-$((_ES_STEPS * _ES_N * _ES_B))}"
ROLLOUTS_PER_STEP=$((TRAIN_BS * GROUP))
case "$BUDGET" in
  gen)    STEPS=$((ES_GENERATIONS / ROLLOUTS_PER_STEP)) ;;
  steps)  STEPS=200 ;;
  manual) STEPS="${STEPS:?BUDGET=manual requires STEPS}" ;;
  *)      echo "unknown BUDGET=$BUDGET (gen|steps|manual)"; exit 2 ;;
esac

GRPO=axis_probe/grpo
# Model-scoped: the parquet stores a chat-template-applied prompt, so an -Instruct parquet and
# a base parquet are different data. Separate dirs let both persist instead of thrashing.
if [[ "$TASK" == "countdown" ]]; then
  DATADIR="$GRPO/data_countdown"          # model-independent: raw context, no chat template
  PARQUET_BUILDER="$GRPO/make_countdown_parquet.py"
  REWARD_PATH="$GRPO/reward_countdown.py"; REWARD_NAME=compute_score
  RAWTMPL="/tmp/rawtmpl_$(basename "$MODEL")"
else
  DATADIR="$GRPO/data_math$SUF"
  PARQUET_BUILDER="$GRPO/make_math_parquet.py"
  REWARD_PATH="$(pwd)/shared_reward.py"; REWARD_NAME=verl_compute_score
fi
OUT=axis_probe/results/grpo$SUF
LOG=axis_probe/logs/grpo$SUF
TAG="grpo_${TASK}_${BUDGET}${STEPS}${SUF}"
# Stable name, NOT $$: a PID-derived path changes every invocation, so the stage-2 skip below
# would never fire and a killed run could not resume. FSDP shards are large, so keep them off
# the 22GB JuiceFS home quota.
CKPT="${CKPT:-/tmp/grpo_${TASK}_${TAG}}"
mkdir -p "$OUT" "$LOG" "$CKPT"
DRIVER="$LOG/driver.log"

echo "[$(date +%F_%T)] start: BUDGET=$BUDGET STEPS=$STEPS rollouts/step=$ROLLOUTS_PER_STEP " \
     "total_gen=$((STEPS * ROLLOUTS_PER_STEP)) (ES arm = $ES_GENERATIONS) maxresp=$MAXRESP" | tee "$DRIVER"

# ---- stage 1: parquet from the SAME make_split the ES arms use ----
# "File exists" is NOT a sufficient check here: the 2026-07-25 parquet existed for weeks after
# the extractor that produced it was found broken. Rebuild unless the stamp matches the
# extractor and config in use right now.
#
# The stamp must cover the MODEL as well as the extractor. make_math_parquet.py stores the
# finished chat-template-applied prompt string, built with C.MODEL's tokenizer, so the parquet
# is model-specific. Checking only the extractor md5 would silently reuse an -Instruct-built
# parquet for a base-model run -- the same stale-artifact failure this check exists to prevent.
NEED_BUILD=1
if [[ "$TASK" == "countdown" ]]; then
  # The countdown parquet carries the raw `context` verbatim: it depends on neither the model
  # nor the answer extractor, so the provenance stamp those guard against does not apply here.
  [[ -f "$DATADIR/train.parquet" ]] && NEED_BUILD=0
elif [[ -f "$DATADIR/provenance.json" ]]; then
  if $P4_PY - "$DATADIR/provenance.json" <<'PROV'
import hashlib, json, os, sys
sys.path.insert(0, "axis_probe/src")
import config as C
prov = json.load(open(sys.argv[1]))
ae = os.path.join(os.path.dirname(os.getcwd()), "ood_eval", "answer_extraction.py")
cur = hashlib.md5(open(ae, "rb").read()).hexdigest()
stale = []
if prov.get("answer_extraction_md5") != cur:
    stale.append("extractor")
if prov.get("model") != C.MODEL:
    stale.append(f"model ({prov.get('model')} -> {C.MODEL})")
if stale:
    print("  parquet stale: " + ", ".join(stale))
sys.exit(1 if stale else 0)
PROV
  then NEED_BUILD=0; fi
fi
if [[ $NEED_BUILD -eq 1 ]]; then
  echo "[$(date +%F_%T)] STAGE1 build parquet (missing or stale extractor stamp)" | tee -a "$DRIVER"
  if [[ "$TASK" == "countdown" ]]; then
    $P4_PY "$PARQUET_BUILDER" "$DATADIR" 2>&1 | tee -a "$DRIVER"
  else
    $P4_PY "$PARQUET_BUILDER" --out_dir "$DATADIR" 2>&1 | tee -a "$DRIVER"
  fi
else
  echo "[$(date +%F_%T)] STAGE1 skip (provenance matches current extractor)" | tee -a "$DRIVER"
  $P4_PY -c "import json;p=json.load(open('$DATADIR/provenance.json'));print('  ',json.dumps(p))" | tee -a "$DRIVER"
fi

# ---- stage 2: verl GRPO ----
if [[ -d "$CKPT/global_step_$STEPS/actor" ]]; then
  echo "[$(date +%F_%T)] STAGE2 skip (checkpoint exists)" | tee -a "$DRIVER"
  TRAIN_S=0
else
  TRAIN_MODEL="$MODEL"
  if [[ "$TASK" == "countdown" ]]; then
    [[ -f "$RAWTMPL/tokenizer_config.orig.json" ]] \
      || $P4_PY "$GRPO/make_raw_template_model.py" "$MODEL" "$RAWTMPL" 2>&1 | tee -a "$DRIVER"
    TRAIN_MODEL="$RAWTMPL"
  fi
  echo "[$(date +%F_%T)] STAGE2 train (task=$TASK, model=$TRAIN_MODEL)" | tee -a "$DRIVER"
  T0=$SECONDS
  CUDA_VISIBLE_DEVICES=$GPU $P4_PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATADIR/train.parquet" \
    data.val_files="$DATADIR/val.parquet" \
    data.train_batch_size=$TRAIN_BS \
    data.max_prompt_length=$MAXPROMPT data.max_response_length=$MAXRESP \
    data.reward_fn_key=data_source data.filter_overlong_prompts=True data.truncation=right \
    actor_rollout_ref.model.path="$TRAIN_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$GROUP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=$REWARD_NAME \
    trainer.logger=[console] trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.default_local_dir="$CKPT" \
    trainer.save_freq=$STEPS trainer.test_freq=-1 trainer.val_before_train=False \
    trainer.total_epochs=1000 trainer.total_training_steps=$STEPS \
    > "$LOG/${TAG}_train.log" 2>&1
  RC=$?
  TRAIN_S=$((SECONDS - T0))
  echo "[$(date +%F_%T)] STAGE2 rc=$RC elapsed=${TRAIN_S}s" | tee -a "$DRIVER"
  if [[ $RC -ne 0 ]]; then
    echo "  !! training failed, see $LOG/${TAG}_train.log" | tee -a "$DRIVER"; exit $RC
  fi
fi

# ---- stage 3: merge + the ES battery ----
echo "[$(date +%F_%T)] STAGE3 merge + eval" | tee -a "$DRIVER"
$P4_PY "$GRPO/eval_grpo.py" \
    --actor_dir "$CKPT/global_step_$STEPS/actor" \
    --out_prefix "$OUT/$TAG" --eval_cap "$EVAL_CAP" --max_tokens "$MAXRESP" --gpu "$GPU" --kl \
    $([[ "$TASK" == "countdown" ]] && echo "--include_countdown --restore_tokenizer $RAWTMPL/tokenizer_config.orig.json") \
    --train_seconds "$TRAIN_S" --train_steps "$STEPS" \
    --rollouts_per_step "$ROLLOUTS_PER_STEP" \
    > "$LOG/${TAG}_eval.log" 2>&1
RC3=$?   # capture BEFORE anything else runs: $(date ...) in the echo would reset $? to 0,
         # which is exactly how a crashed stage 3 got logged as rc=0 on 2026-08-15.
echo "[$(date +%F_%T)] STAGE3 rc=$RC3" | tee -a "$DRIVER"
if [[ $RC3 -ne 0 ]]; then
  echo "  !! eval failed -- NO summary written. See $LOG/${TAG}_eval.log" | tee -a "$DRIVER"
  exit $RC3
fi
echo "[$(date +%F_%T)] COMPLETE -> $OUT/${TAG}_summary.json" | tee -a "$DRIVER"
echo "  checkpoints left in $CKPT (delete when done; FSDP shards are large)" | tee -a "$DRIVER"
echo "  compare with:  \$P4_PY axis_probe/compare_all.py" | tee -a "$DRIVER"
