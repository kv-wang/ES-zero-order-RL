#!/usr/bin/env bash
# Countdown battery (2026-07-29, user-requested): TRAIN on countdown, EVAL OOD on math sets.
# Direction inverted vs every earlier arm (those trained math, probed countdown OOD).
#
# Three arms, matched STEPS (100) -- NOT wall-clock matched (per user request); wall-clock and
# s/step are recorded per arm. N=30 B=100 for both ES arms; GRPO at its own established batch
# geometry (TRAIN_BS=16 x GROUP=8 = 128 rollouts/step, same as every prior GRPO run here).
#   arm1 loraes : LoRA-ES rank16 ALL 7 targets (attn+mlp), VANILLA sampling (non-antithetic --
#                 the regime that showed split-half rho=.46), sigma*=0.015 alpha=0.0075
#                 (flip-matched for exactly this adapter config in the overnight campaign)
#                 -> eval_oldproto lambda {1.0, 0.5} + perq
#   arm2 fulles : full-param ES sigma=1e-3 alpha=5e-4 (repo reference scale, all prior 3B
#                 full-param arms) -> in-process final eval + KL proxy + perq
#   arm3 grpo   : verl GRPO 100 steps, raw-context prompt parity via the identity-chat-template
#                 model copy (grpo_ood/make_raw_template_model.py) -> merge -> RESTORE the real
#                 tokenizer into hf_merged (math evals need the true chat template!) -> eval + perq
#
# Protocol: max_tokens 2048 for ALL training and ALL eval (never mix caps); engine util matched
# at TRAIN_GPU_MEM_UTIL=0.50 for training, EVAL_GPU_MEM_UTIL=0.60 for eval (config.py).
# Storage: ckpts + merged HF in /tmp (bulky, disposable), logs/results/theta in home (durable).
# Stages are resumable (skip when output exists); a failed arm does NOT block later arms.
set -uo pipefail
cd "$(dirname "$0")"
TRACK_B=$(pwd)
ES_BENCH=$(cd .. && pwd)
source "$ES_BENCH/track_a/env.sh"

GPU=${GPU:-0}
MODEL=Qwen/Qwen2.5-3B-Instruct
STEPS=${STEPS:-100}; N=${N:-30}; B=${B:-100}; MAXTOK=2048
TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_SIGMA=0.015; LORA_ALPHA=0.0075
FP_SIGMA=1e-3;    FP_ALPHA=5e-4
RES=$TRACK_B/results/countdown_battery
LOGS=$TRACK_B/logs/countdown_battery
mkdir -p "$RES/loraes" "$RES/fulles" "$RES/grpo" "$LOGS"     # mkdir BEFORE any redirect
STATUS=$RES/STATUS.txt
stage() { echo "[$(date +%F' '%T)] $*" | tee -a "$STATUS"; }

wait_gpu() {  # vLLM/ray teardown holds VRAM after the process exits
  for i in $(seq 1 90); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
    [ "${U:-99999}" -lt 2000 ] && { stage "GPU drained (${U}MiB)"; return 0; }
    sleep 10
  done
  stage "WARN: GPU still at ${U}MiB after 15min"
}

stage "=== countdown battery start: steps=$STEPS N=$N B=$B maxtok=$MAXTOK gpu=$GPU"

# ================= arm 1: LoRA-ES rank16 full-target =========================================
LORAES_CKPT=/tmp/countdown_loraes_ckpt
if [ -f "$RES/loraes/ood_eval.json" ]; then
  stage "arm1 loraes  SKIP (ood_eval.json present)"
else
  if [ ! -f "$RES/loraes/arm_summary.json" ]; then
    stage "arm1 loraes  TRAIN start (N$N B$B r16 all-targets vanilla sig=$LORA_SIGMA)"
    $P4_PY src/es_lora_main.py --model "$MODEL" --dataset countdown \
      --population_size "$N" --batch "$B" --num_steps "$STEPS" \
      --sigma "$LORA_SIGMA" --alpha "$LORA_ALPHA" --rank 16 --targets "$TARGETS" \
      --max_tokens "$MAXTOK" --chunk 1500 --pop_seed 0 --data_seed 42 \
      --ckpt_every 25 --probe_every 25 --no_gates --gpu "$GPU" \
      --ckpt_dir "$LORAES_CKPT" --out_prefix "$RES/loraes/arm" \
      > "$LOGS/loraes_train.log" 2>&1
    rc=$?
    [ -f "$RES/loraes/arm_summary.json" ] || stage "arm1 loraes  TRAIN FAILED rc=$rc (see $LOGS/loraes_train.log)"
  fi
  if [ -f "$RES/loraes/arm_summary.json" ]; then
    FINAL=$($P4_PY -c "
import json, os
s = json.load(open('$RES/loraes/arm_summary.json'))
c = [x for x in s['checkpoints'] if os.path.exists(x['path'])]
print(c[-1]['path'] if c else '')")
    if [ -n "$FINAL" ]; then
      cp "$FINAL" "$RES/loraes/theta_final.pt"          # durable copy (~120MB, home has room)
      wait_gpu
      stage "arm1 loraes  EVAL start (theta=$FINAL, lambda 1.0+0.5, perq)"
      $P4_PY src/eval_oldproto.py --model "$MODEL" --theta "$RES/loraes/theta_final.pt" \
        --rank 16 --targets "$TARGETS" --lambdas 1.0,0.5 --tag base \
        --eval_cap 300 --max_tokens 2048 --perq --gpu "$GPU" \
        --out "$RES/loraes/ood_eval.json" > "$LOGS/loraes_eval.log" 2>&1 \
        && stage "arm1 loraes  EVAL done" \
        || stage "arm1 loraes  EVAL FAILED (see $LOGS/loraes_eval.log)"
    else
      stage "arm1 loraes  no surviving checkpoint -- /tmp wiped? rerun to retrain"
    fi
  fi
fi
wait_gpu

# ================= arm 2: full-param ES ======================================================
FULLES_CKPT=/tmp/countdown_fulles_ckpt
if [ -f "$RES/fulles/arm_summary.json" ]; then
  stage "arm2 fulles  SKIP (arm_summary.json present)"
else
  stage "arm2 fulles  TRAIN start (N$N B$B sig=$FP_SIGMA alpha=$FP_ALPHA, in-process eval+KL)"
  mkdir -p "$FULLES_CKPT"
  $P4_PY src/es_train_fullparam.py --reward binary --model "$MODEL" --dataset countdown \
    --population_size "$N" --batch "$B" --num_steps "$STEPS" \
    --sigma "$FP_SIGMA" --alpha "$FP_ALPHA" --max_tokens "$MAXTOK" \
    --pop_seed 0 --gpu "$GPU" --kl --eval_final --eval_cap 300 \
    --ckpt_dir "$FULLES_CKPT" --out_prefix "$RES/fulles/arm" \
    > "$LOGS/fulles_train.log" 2>&1
  [ -f "$RES/fulles/arm_summary.json" ] \
    && stage "arm2 fulles  DONE (summary + in-process eval written)" \
    || stage "arm2 fulles  FAILED rc=$? (see $LOGS/fulles_train.log)"
fi
wait_gpu

# ================= arm 3: GRPO ===============================================================
GRPO_EXP=grpo_full_3b_countdown
GRPO_CKPT=/tmp/grpo_ckpt_$GRPO_EXP
ACTOR=$GRPO_CKPT/global_step_$STEPS/actor
HF_MERGED=$GRPO_CKPT/global_step_$STEPS/hf_merged
RAWTMPL=/tmp/qwen3b_rawtmpl
if [ -f "$RES/grpo/ood_eval.json" ]; then
  stage "arm3 grpo    SKIP (ood_eval.json present)"
else
  FREE_TMP=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
  if [ "${FREE_TMP:-0}" -lt 40 ]; then
    stage "arm3 grpo    ABORT: /tmp only ${FREE_TMP}G free, need >=40G"
  else
    [ -f "$RAWTMPL/tokenizer_config.orig.json" ] \
      || $P4_PY grpo_ood/make_raw_template_model.py "$MODEL" "$RAWTMPL" >> "$LOGS/grpo_prep.log" 2>&1
    [ -f "$TRACK_B/grpo_ood/data_countdown/train.parquet" ] \
      || $P4_PY grpo_ood/make_countdown_parquet.py "$TRACK_B/grpo_ood/data_countdown" >> "$LOGS/grpo_prep.log" 2>&1
    if [ ! -f "$ACTOR/model_world_size_1_rank_0.pt" ]; then
      stage "arm3 grpo    TRAIN start ($STEPS steps, TRAIN_BS=16 GROUP=8, raw-template model)"
      MODEL=$RAWTMPL DATA_DIR=$TRACK_B/grpo_ood/data_countdown GPU=$GPU LORA_RANK=0 \
      STEPS=$STEPS TRAIN_BS=16 GROUP=8 MICRO=8 MAXRESP=$MAXTOK MAXPROMPT=1024 \
      EXP=$GRPO_EXP REWARD_FN=$TRACK_B/grpo_ood/reward_countdown.py \
      bash "$TRACK_B/grpo_ood/run_grpo_train.sh" > "$LOGS/grpo_train.log" 2>&1
      # trailing echo masks verl's exit code (known gotcha) -- trust the artifact
    fi
    if [ -f "$ACTOR/model_world_size_1_rank_0.pt" ]; then
      wait_gpu
      if [ ! -f "$HF_MERGED/config.json" ]; then
        stage "arm3 grpo    MERGE fsdp->HF"
        $P4_PY -m verl.model_merger merge --backend fsdp \
          --local_dir "$ACTOR" --target_dir "$HF_MERGED" > "$LOGS/grpo_merge.log" 2>&1
      fi
      if [ -f "$HF_MERGED/config.json" ]; then
        # CRITICAL: hf_merged inherited the identity chat template from the raw-template model.
        # Math-set eval prompts are built with apply_chat_template -- restore the real one.
        cp "$RAWTMPL/tokenizer_config.orig.json" "$HF_MERGED/tokenizer_config.json"
        stage "arm3 grpo    tokenizer restored; EVAL start (cap300/2048, perq)"
        $P4_PY grpo_ood/eval_grpo_ood.py --actor_dir "$ACTOR" --tag grpo_countdown \
          --out "$RES/grpo/ood_eval.json" --eval_cap 300 --perq --gpu "$GPU" \
          > "$LOGS/grpo_eval.log" 2>&1 \
          && stage "arm3 grpo    EVAL done" \
          || stage "arm3 grpo    EVAL FAILED (see $LOGS/grpo_eval.log)"
      else
        stage "arm3 grpo    MERGE FAILED (see $LOGS/grpo_merge.log)"
      fi
    else
      stage "arm3 grpo    TRAIN produced no checkpoint (see $LOGS/grpo_train.log)"
    fi
  fi
fi
wait_gpu

# ================= comparison ================================================================
stage "comparison"
$P4_PY - "$RES" "$TRACK_B/results/exp3b_math/base_3b.json" <<'PY' 2>&1 | tee -a "$STATUS"
import json, os, sys
res_dir, base_path = sys.argv[1:3]
OODSETS = ["math500", "gsm8k", "svamp", "minerva_math", "olympiadbench"]  # countdown = ID here

def row(ev, src):
    if ev is None:
        return None
    r = {"source": src, "countdown_ID": ev.get("countdown", {}).get("accuracy")}
    for k in OODSETS:
        r[k] = ev.get(k, {}).get("accuracy")
    r["L3_5"] = ev.get("math500", {}).get("primary_L3_5_accuracy")
    vals = [r[k] for k in OODSETS if r.get(k) is not None]
    r["OODavg_math5"] = round(sum(vals) / len(vals), 4) if len(vals) == len(OODSETS) else None
    return r

out = {"note": "trained on countdown[300:], eval countdown[:300]=ID + math sets=OOD; "
               "cap300/2048 greedy; steps matched (100), NOT wall-clock matched", "rows": {}}
try:
    out["rows"]["base_stored"] = row(json.load(open(base_path))["eval"], base_path)
except Exception as e:
    out["rows"]["base_stored"] = f"unreadable: {e}"
p = os.path.join(res_dir, "loraes", "ood_eval.json")
if os.path.exists(p):
    ent = json.load(open(p))["entries"]
    for tag, ev in ent.items():
        out["rows"][f"loraes:{tag}"] = row(ev, p)
p = os.path.join(res_dir, "fulles", "arm_summary.json")
if os.path.exists(p):
    s = json.load(open(p))
    out["rows"]["fulles"] = row(s.get("eval_final"), p)
    if out["rows"]["fulles"] is not None:
        out["rows"]["fulles"]["kl_proxy"] = s.get("kl_proxy_drift")
        out["rows"]["fulles"]["s_per_step"] = s.get("s_per_step_mean")
p = os.path.join(res_dir, "grpo", "ood_eval.json")
if os.path.exists(p):
    out["rows"]["grpo"] = row(json.load(open(p))["eval"], p)
json.dump(out, open(os.path.join(res_dir, "COMPARISON.json"), "w"), indent=2)
print("COMPARISON:\n" + json.dumps(out["rows"], indent=2))
PY

stage "=== countdown battery COMPLETE"
{ echo "## $(date +%F) countdown battery: LoRA-ES r16-full + full-param ES + GRPO, 100 steps,"
  echo "##   train countdown[300:] B100 N30, eval countdown[:300] ID + math OOD, cap300/2048"
} >> "$TRACK_B/commands.log"
