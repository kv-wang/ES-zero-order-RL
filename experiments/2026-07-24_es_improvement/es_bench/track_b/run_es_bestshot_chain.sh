#!/usr/bin/env bash
# Best-shot LoRA-ES vs GRPO at matched wall-clock -- fully unattended pipeline.
#
# GOAL. Give LoRA-ES its fairest measured configuration (every lever from
# LORAES_WALLCLOCK_ANALYSIS.md) and compare against GRPO@2048 with train budget, eval protocol,
# and engine settings all matched:
#   - B=100            (fidelity/sqrt(cost) optimum, sec.2)
#   - attn-only rank 8 (d 29.9M -> 3.69M, free in generation time, sec.4; validated 2026-07-28)
#   - antithetic pairs (variance halved at identical cost, sec.5; implemented 2026-07-28)
#   - sigma* re-selected by the Phase A flip-rate procedure under the NEW d (sec.7 requirement)
#   - alpha = sigma*/2 (pre-registered Phase A convention)
#   - train cap 2048 = GRPO's max_response_length = eval max_tokens (protocol-trap lesson)
#   - engine util = TRAIN_GPU_MEM_UTIL for both methods (config lock 2026-07-28)
#   - wall-clock budget = GRPO's own measured train time (sum of verl timing_s/step)
#
# STAGES (each skipped if its output exists => resumable):
#   0 wait     : block until the GRPO chain's summary exists and the GPU has drained
#   1 sigma    : LoRA flip-rate grid under r8 attn-only; select sigma* vs EXISTING 3B reference
#                (the full-param reference does not depend on the adapter config -- reused)
#   2 timing   : 5 timed steps of the exact arm config -> s/step at util 0.50
#   3 arm      : es_lora_main with --deadline_ts = now + GRPO wall; gates ON (a gate exit is a
#                recorded negative result, not a failure to retry)
#   4 eval     : final theta at cap300/2048 via eval_oldproto (lambda 1.0 and 0.5)
#   5 compare  : one JSON row set vs GRPO chain + stored base_3b.json
set -uo pipefail
cd "$(dirname "$0")"; TRACK_B=$(pwd); ES_BENCH=$(cd .. && pwd)
source "$ES_BENCH/track_a/env.sh"

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
GPU=${GPU:-0}
GRPO_EXP=${GRPO_EXP:-grpo_full_3b_math_t2048}
EXP=${EXP:-loraes_bestshot_3b_math}
N=${N:-30}; B=${B:-100}
RANK=${RANK:-8}; TARGETS=${TARGETS:-q_proj,k_proj,v_proj,o_proj}
MAXTOK=${MAXTOK:-2048}
POP_SEED=${POP_SEED:-0}; DATA_SEED=${DATA_SEED:-42}
SIGMA_GRID=${SIGMA_GRID:-1.5e-2,3e-2,5e-2,9e-2}   # brackets norm-equivalents of the r16 curve

RES=$TRACK_B/results/$EXP; LOGS=$TRACK_B/logs/$EXP
PHA=$RES/phaseA_r8
mkdir -p "$RES" "$LOGS" "$PHA"
STATUS=$RES/STATUS.txt
stage(){ echo "[$(date +%F' '%T)] $*" | tee -a "$STATUS"; }
fail(){ stage "FAILED: $*"; exit 1; }

GRPO_SUMMARY=$TRACK_B/results/$GRPO_EXP/summary.json
GRPO_TRAINLOG=$TRACK_B/logs/$GRPO_EXP/train.log

stage "=== bestshot chain start: $EXP  N=$N B=$B r$RANK targets=$TARGETS cap=$MAXTOK antithetic"

# ---- stage 0: wait for the GRPO reference arm ------------------------------------------------
stage "stage0 wait for $GRPO_SUMMARY (poll 120s)"
for i in $(seq 1 200); do [ -f "$GRPO_SUMMARY" ] && break; sleep 120; done
[ -f "$GRPO_SUMMARY" ] || fail "GRPO chain never finished (waited ~6.6h)"
for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1)
  [ "${USED:-99999}" -lt 5000 ] && break; sleep 10
done
stage "stage0 done: GRPO summary present, GPU at ${USED}MiB"

# GRPO pure-train wall-clock = sum of verl per-step times (excludes engine startup; the ES
# deadline likewise starts after vLLM startup, so the budgets compare like for like)
GRPO_WALL=$($P4_PY - "$GRPO_TRAINLOG" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
print(int(sum(float(x) for x in re.findall(r"timing_s/step:([\d.]+)", txt))))
PY
) || fail "could not parse GRPO wall from $GRPO_TRAINLOG"
[ "$GRPO_WALL" -gt 600 ] || fail "GRPO wall parsed as ${GRPO_WALL}s -- implausible"
stage "GRPO train wall-clock budget: ${GRPO_WALL}s"

# ---- stage 1: sigma* under the reduced-d template --------------------------------------------
if [ -f "$PHA/SIGMA_STAR.json" ]; then
  stage "stage1 sigma  SKIP (already selected)"
else
  stage "stage1 sigma  START (lora grid $SIGMA_GRID, r$RANK attn-only, refs reused)"
  cp "$TRACK_B/results/overnight/phaseA/ref_math.json" \
     "$TRACK_B/results/overnight/phaseA/ref_gsm8k.json" "$PHA/"
  for DS in math gsm8k; do
    $P4_PY "$TRACK_B/src/probe_sigma_scale.py" --arm lora --model "$MODEL" --dataset $DS \
      --n_prompts 16 --n_seeds 8 --sigmas "$SIGMA_GRID" --rank "$RANK" --targets "$TARGETS" \
      --gpu "$GPU" --out "$PHA/lora_${DS}.json" > "$LOGS/sigma_${DS}.log" 2>&1 \
      || fail "sigma probe $DS, see $LOGS/sigma_${DS}.log"
  done
  $P4_PY "$TRACK_B/src/phaseA_select.py" --dir "$PHA" --out "$PHA/SIGMA_STAR.json" \
    >> "$STATUS" 2>&1 || fail "phaseA_select"
fi
SIGMA=$($P4_PY -c "import json;print(json.load(open('$PHA/SIGMA_STAR.json'))['sigma_star'])")
ALPHA=$($P4_PY -c "import json;print(json.load(open('$PHA/SIGMA_STAR.json'))['alpha'])")
stage "stage1 done: sigma*=$SIGMA alpha=$ALPHA"

# ---- stage 2: s/step at the matched engine setting -------------------------------------------
TIMING=$RES/timing_5step_summary.json
if [ -f "$TIMING" ]; then
  stage "stage2 timing SKIP"
else
  stage "stage2 timing START (5 steps, exact arm config)"
  $P4_PY "$TRACK_B/src/es_lora_main.py" --model "$MODEL" --dataset math \
    --population_size "$N" --batch "$B" --num_steps 5 --sigma "$SIGMA" --alpha "$ALPHA" \
    --rank "$RANK" --targets "$TARGETS" --antithetic --max_tokens "$MAXTOK" \
    --pop_seed "$POP_SEED" --data_seed "$DATA_SEED" --no_gates --gpu "$GPU" \
    --ckpt_every 1000 --probe_every 1000 \
    --out_prefix "$RES/timing_5step" > "$LOGS/timing.log" 2>&1 \
    || fail "timing bench, see $LOGS/timing.log"
fi
SPS=$($P4_PY -c "import json;print(round(json.load(open('$TIMING'))['s_per_step_mean'],2))")
EST_STEPS=$($P4_PY -c "print(int($GRPO_WALL/$SPS))")
stage "stage2 done: $SPS s/step at util-matched setting -> ~$EST_STEPS steps in budget"

# ---- stage 3: the matched-budget arm ---------------------------------------------------------
ARM=$RES/arm
if [ -f "${ARM}_summary.json" ]; then
  stage "stage3 arm    SKIP"
else
  stage "stage3 arm    START (deadline = now + ${GRPO_WALL}s; gates ON)"
  DEADLINE=$($P4_PY -c "import time;print(time.time()+$GRPO_WALL)")
  $P4_PY "$TRACK_B/src/es_lora_main.py" --model "$MODEL" --dataset math \
    --population_size "$N" --batch "$B" --num_steps 2000 --sigma "$SIGMA" --alpha "$ALPHA" \
    --rank "$RANK" --targets "$TARGETS" --antithetic --max_tokens "$MAXTOK" \
    --pop_seed "$POP_SEED" --data_seed "$DATA_SEED" --deadline_ts "$DEADLINE" \
    --ckpt_every 25 --probe_every 25 --gpu "$GPU" \
    --ckpt_dir "$RES/ckpt" \
    --out_prefix "$ARM" > "$LOGS/arm.log" 2>&1
  RC=$?
  case $RC in
    0)  stage "stage3 arm completed/deadline" ;;
    90) stage "stage3 arm STOPPED BY STEP-GATE (no fitness slope AND no ranking signal) -- recorded as negative" ;;
    91) stage "stage3 arm STOPPED BY KL-GUARD (policy destroyed early) -- recorded as negative" ;;
    *)  fail "arm crashed rc=$RC, see $LOGS/arm.log" ;;
  esac
  [ -f "${ARM}_summary.json" ] || fail "arm wrote no summary"
fi

# ---- stage 4: eval the final theta at the stored-baseline protocol ---------------------------
EVAL=$RES/ood_eval.json
FINAL_CKPT=$($P4_PY -c "
import json,glob,os
s=json.load(open('${ARM}_summary.json'))
cands=[c for c in s['checkpoints'] if os.path.exists(c['path'])]
print(cands[-1]['path'] if cands else '')")
[ -n "$FINAL_CKPT" ] || fail "no surviving checkpoint (container restart wiped /tmp?) -- rerun stage3"
stage "evaluating theta: $FINAL_CKPT"
cp "$FINAL_CKPT" "$RES/theta_final.pt"    # canonical final-theta path (~15MB, already durable)
if [ -f "$EVAL" ]; then
  stage "stage4 eval   SKIP"
else
  stage "stage4 eval   START (cap300/2048, lambda 1.0 + 0.5)"
  $P4_PY "$TRACK_B/src/eval_oldproto.py" --model "$MODEL" --theta "$RES/theta_final.pt" \
    --rank "$RANK" --targets "$TARGETS" --lambdas 1.0,0.5 --tag "$EXP" \
    --eval_cap 300 --max_tokens 2048 --gpu "$GPU" --out "$EVAL" \
    > "$LOGS/eval.log" 2>&1 || fail "eval, see $LOGS/eval.log"
fi

# ---- stage 5: the comparison row -------------------------------------------------------------
stage "stage5 compare"
$P4_PY - "$RES" "$GRPO_SUMMARY" "$EVAL" "${ARM}_summary.json" "$TIMING" "$GRPO_WALL" <<'PY' \
  2>&1 | tee -a "$STATUS"
import json, statistics, sys
res_dir, grpo_summary, es_eval, arm_summary, timing, grpo_wall = sys.argv[1:7]
OOD = ["math500", "svamp", "minerva_math", "olympiadbench", "countdown"]

def row_from_eval(ev):
    acc = {k: v.get("accuracy") for k, v in ev.items()}
    ood = [acc[k] for k in OOD if acc.get(k) is not None]
    return {"OODavg": round(statistics.mean(ood), 4) if ood else None, "OOD_n": len(ood),
            "L3_5": ev.get("math500", {}).get("primary_L3_5_accuracy"), **acc}

grpo = json.load(open(grpo_summary))
es = json.load(open(es_eval))["entries"]
arm = json.load(open(arm_summary))
out = {"budget_s": int(grpo_wall),
       "grpo": grpo["row"],
       "es_arm": {k: arm[k] for k in ["population_size", "batch", "rank", "targets", "antithetic",
                                      "sigma", "alpha", "steps_completed", "wall_clock_s",
                                      "s_per_step_mean", "stop_reason", "fit_first20",
                                      "fit_last20", "median_split_half_spearman"]},
       "rows": {tag: row_from_eval(ev) for tag, ev in es.items()}}
json.dump(out, open(f"{res_dir}/COMPARISON.json", "w"), indent=2)
print("COMPARISON " + json.dumps({k: v for k, v in out["rows"].items()}, indent=2))
print(f"ES arm: {arm['steps_completed']} steps in {arm['wall_clock_s']:.0f}s "
      f"({arm.get('s_per_step_mean') or 0:.1f}s/step), stop: {arm['stop_reason']}")
PY

{ echo "## $(date +%F) LoRA-ES bestshot chain $EXP: sigma-recal(r$RANK attn-only) -> timing -> "
  echo "##   wall-matched arm (N$N B$B antithetic cap$MAXTOK, budget=GRPO ${GRPO_WALL}s) -> OOD eval -> COMPARISON.json"
} >> "$TRACK_B/commands.log"
stage "=== bestshot chain COMPLETE ==="
