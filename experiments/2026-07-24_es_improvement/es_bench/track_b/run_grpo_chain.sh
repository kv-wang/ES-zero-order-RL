#!/usr/bin/env bash
# Autonomous GRPO chain:  train -> merge -> OOD eval -> durable results.  No human between stages.
#
# WHY THIS EXISTS.  On 2026-07-28 the t2048 run finished and wrote its checkpoint to /tmp at
# 17:39:45; the container died at 17:47 and took the checkpoint with it.  1h43m of training was
# lost not because /tmp is the wrong place -- it is the right place for bulky, short-lived
# optimizer state -- but because advancing from "saved" to "evaluated" required a person to be
# connected.  This script removes the person.  Exposure shrinks from "until someone reconnects"
# to "the ~20 min the eval itself takes".
#
# STORAGE CONTRACT (deliberate, do not "fix" by moving checkpoints to home):
#   /tmp (overlay, 198G free, dies with the container) : verl checkpoint + merged HF model.
#                                                        Bulky, reconstructible, short-lived.
#   home (JuiceFS, 27G free, survives)                 : logs, results JSON, summary CSV.
#                                                        Small, irreplaceable.
# Home is a 64G mount that has already been filled once and corrupted a checkpoint -- putting
# ~20GB of optimizer state there is how you lose the next run instead of this one.
#
# RESUMABLE.  Re-running skips any stage whose output already exists, so if this dies at eval you
# relaunch and it evaluates the existing checkpoint instead of retraining.
#
# Usage:  bash run_grpo_chain.sh                       # defaults = the lost t2048 run
#         MAXRESP=1024 EXP=grpo_full_3b_math_t1024 bash run_grpo_chain.sh
set -uo pipefail

cd "$(dirname "$0")"                                   # track_b
TRACK_B=$(pwd)
ES_BENCH=$(cd .. && pwd)

# ---- config (defaults reproduce the run lost on 2026-07-28, recovered from its train.log) ----
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
EXP=${EXP:-grpo_full_3b_math_t2048}
STEPS=${STEPS:-200}
MAXRESP=${MAXRESP:-2048}
MAXPROMPT=${MAXPROMPT:-1024}
TRAIN_BS=${TRAIN_BS:-16}
GROUP=${GROUP:-8}
MICRO=${MICRO:-8}
LORA_RANK=${LORA_RANK:-0}
GPU=${GPU:-0}
EVAL_CAP=${EVAL_CAP:-300}
EVAL_MAXTOK=${EVAL_MAXTOK:-2048}
DATA_DIR=${DATA_DIR:-$TRACK_B/grpo_ood/data_math}
PRUNE_CKPT=${PRUNE_CKPT:-0}                            # 1 = delete /tmp ckpt after a good eval

CKPT=/tmp/grpo_ckpt_$EXP                               # bulky, disposable
ACTOR=$CKPT/global_step_$STEPS/actor
HF_MERGED=$CKPT/global_step_$STEPS/hf_merged
RESULTS=$TRACK_B/results/$EXP                          # durable
LOGS=$TRACK_B/logs/$EXP
mkdir -p "$RESULTS" "$LOGS" "$CKPT"                    # mkdir BEFORE any redirect into them
STATUS=$RESULTS/STATUS.txt

source "$ES_BENCH/track_a/env.sh"

stage() { echo "[$(date +%F' '%T)] $*" | tee -a "$STATUS"; }
fail()  { stage "FAILED: $*"; exit 1; }

stage "=== chain start: exp=$EXP model=$MODEL steps=$STEPS maxresp=$MAXRESP lora_rank=$LORA_RANK"
stage "ckpt(/tmp,disposable)=$CKPT  results(home,durable)=$RESULTS"

# ---- preflight: /tmp must have room for ~20GB of fp32 optimizer state ----
FREE_TMP=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
[ "${FREE_TMP:-0}" -lt 40 ] && fail "/tmp has only ${FREE_TMP}G free, need >=40G for a 3B checkpoint"
stage "preflight ok: /tmp ${FREE_TMP}G free"

# ---- stage 1: train ----------------------------------------------------------------------
if [ -f "$ACTOR/model_world_size_1_rank_0.pt" ]; then
  stage "stage1 train  SKIP (checkpoint already present at $ACTOR)"
else
  stage "stage1 train  START (~1.7h at maxresp=2048)"
  MODEL=$MODEL DATA_DIR=$DATA_DIR GPU=$GPU LORA_RANK=$LORA_RANK STEPS=$STEPS \
  TRAIN_BS=$TRAIN_BS GROUP=$GROUP MICRO=$MICRO MAXRESP=$MAXRESP MAXPROMPT=$MAXPROMPT \
  EXP=$EXP bash "$TRACK_B/grpo_ood/run_grpo_train.sh" > "$LOGS/train.log" 2>&1
  # run_grpo_train.sh ends in `echo "peak VRAM..."`, which masks verl's exit code (known gotcha),
  # so trust the artifact rather than $?.
  [ -f "$ACTOR/model_world_size_1_rank_0.pt" ] || fail "training produced no checkpoint at $ACTOR"
  stage "stage1 train  DONE -> $ACTOR"
fi

# ---- wait for the GPU: ray/vLLM teardown holds VRAM for a while after the process returns ----
stage "waiting for GPU $GPU to drain before eval"
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
  [ "${USED:-99999}" -lt 5000 ] && break
  sleep 10
done
stage "GPU $GPU at ${USED}MiB, proceeding"

# ---- stage 2+3: merge (fsdp -> HF) + OOD eval.  eval_grpo_ood.py skips the merge if hf_merged
#      already exists, so this stage is itself resumable. -----------------------------------
EVAL_JSON=$RESULTS/ood_eval.json
if [ -f "$EVAL_JSON" ]; then
  stage "stage2+3 merge+eval  SKIP ($EVAL_JSON already written)"
else
  stage "stage2+3 merge+eval  START (cap=$EVAL_CAP max_tokens=$EVAL_MAXTOK)"
  $P4_PY "$TRACK_B/grpo_ood/eval_grpo_ood.py" \
      --actor_dir "$ACTOR" --tag "$EXP" --out "$EVAL_JSON" \
      --eval_cap "$EVAL_CAP" --gpu "$GPU" > "$LOGS/eval.log" 2>&1 \
    || fail "merge+eval failed, see $LOGS/eval.log (checkpoint is intact, relaunch to retry)"
  [ -f "$EVAL_JSON" ] || fail "eval wrote no results"
  stage "stage2+3 merge+eval  DONE -> $EVAL_JSON"
fi

# ---- stage 4: durable summary (small, lands in home) ---------------------------------------
stage "stage4 summarize"
$P4_PY - "$EVAL_JSON" "$RESULTS/summary.json" "$TRACK_B/results/grpo_chain_summary.csv" \
        "$EXP" "$MAXRESP" "$EVAL_MAXTOK" "$LOGS/train.log" <<'PY' 2>&1 | tee -a "$STATUS"
import json, os, re, sys, statistics
ev_path, out_json, out_csv, exp, maxresp, evaltok, train_log = sys.argv[1:8]
ev = json.load(open(ev_path))["eval"]
OOD = ["math500", "svamp", "minerva_math", "olympiadbench", "countdown"]
acc = {k: v.get("accuracy") for k, v in ev.items()}
ood = [acc[k] for k in OOD if acc.get(k) is not None]
row = {
    "exp": exp, "train_max_response": int(maxresp), "eval_max_tokens": int(evaltok),
    # OOD_n is recorded because an OODavg over 4 sets is NOT comparable to one over 5 --
    # older runs predate countdown being added to the battery.
    "OODavg": round(statistics.mean(ood), 4) if ood else None,
    "OOD_n": len(ood),
    "L3_5": ev.get("math500", {}).get("primary_L3_5_accuracy"),
    **{k: acc.get(k) for k in ["gsm8k"] + OOD},
}
# training-side context, parsed from the log so the arm is interpretable without it
try:
    txt = open(train_log, errors="ignore").read().replace("\r", "\n")
    steps = [l for l in txt.split("\n") if "training/global_step:" in l]
    def g(l, k):
        m = re.search(re.escape(k) + r":([-\d.e+]+)", l)
        return float(m.group(1)) if m else None
    if steps:
        f10, l10 = steps[:10], steps[-10:]
        row["reward_first10"] = round(statistics.mean([g(l, "critic/score/mean") for l in f10]), 4)
        row["reward_last10"] = round(statistics.mean([g(l, "critic/score/mean") for l in l10]), 4)
        row["resp_len_last10"] = round(statistics.mean([g(l, "response_length/mean") for l in l10]), 1)
        row["clip_ratio_last10"] = round(statistics.mean([g(l, "response_length/clip_ratio") for l in l10]), 4)
        row["s_per_step"] = round(statistics.mean([x for x in (g(l, "timing_s/step") for l in steps) if x]), 2)
        row["n_steps"] = len(steps)
except Exception as e:
    row["train_log_parse_error"] = str(e)
json.dump({"row": row, "eval": ev}, open(out_json, "w"), indent=2)
hdr = not os.path.exists(out_csv)
with open(out_csv, "a") as f:
    if hdr:
        f.write(",".join(row) + "\n")
    f.write(",".join("" if row[k] is None else str(row[k]) for k in row) + "\n")
print("SUMMARY " + json.dumps(row, indent=2))
PY

stage "stage4 summarize DONE -> $RESULTS/summary.json (+ results/grpo_chain_summary.csv)"

# ---- optional prune: only after results are safely in home ---------------------------------
if [ "$PRUNE_CKPT" = "1" ] && [ -f "$RESULTS/summary.json" ]; then
  stage "pruning /tmp checkpoint (PRUNE_CKPT=1), keeping merged HF at $HF_MERGED"
  rm -rf "$CKPT/global_step_$STEPS/actor"
fi

{ echo "## $(date +%F) GRPO chain $EXP: train($STEPS steps, maxresp=$MAXRESP) -> merge -> OOD eval"
  echo "##   cap=$EVAL_CAP max_tokens=$EVAL_MAXTOK; ckpt /tmp (disposable), results $RESULTS (durable)"
} >> "$TRACK_B/commands.log"

stage "=== chain COMPLETE ==="
