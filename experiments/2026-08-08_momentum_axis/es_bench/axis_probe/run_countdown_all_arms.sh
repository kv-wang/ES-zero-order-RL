#!/usr/bin/env bash
# Countdown line: GRPO + the three ES variants on the ONE regime this repo has ever measured
# real ES signal in (2026-07-30: split-half rho = 0.51, base 0.08 -> full-ES 0.450).
#
#   stage 0  untrained base reference   (countdown ID + math battery as OOD)
#   stage 1  GRPO                       -> run_grpo.sh TASK=countdown
#   stage 2  ES vanilla                 -\
#   stage 3  ES baseaxis                 |-> es_train_axis.py --dataset countdown
#   stage 4  ES momentum                -/
#   stage 5  combined readout
#
# WHY THIS IS NOT run_all_arms.sh WITH A FLAG
# -------------------------------------------
# Countdown differs from the math line in four ways that all have to move together:
#   model    Qwen2.5-3B-Instruct, not Math-1.5B. The paper's countdown experiments use the
#            Qwen2.5-*-Instruct family and 3B is what 07-30 measured rho=0.51 on.
#   prompt   the dataset's own raw `context` (opens with <think>), NO chat template. GRPO
#            therefore trains against an identity-chat-template model copy so verl's rollout
#            string is char-identical to the ES arms'; the real tokenizer is restored into the
#            merged HF dir before the math OOD sets are scored.
#   reward   countdown_task.answer_reward_function, shared by the ES trainer, the verl reward
#            and eval_core._eval_countdown. It never touches ood_eval/answer_extraction, which
#            is why the 2026-08 \boxed defect never contaminated countdown numbers.
#   budget   N=30 B=100 steps=100 (the 07-30 battery), NOT the math line's N=16 B=8 steps=200.
#            B=100 is the point: split-half rho measures whether B is large enough to rank
#            members, and B=8 gives fitness only ~3 distinct values (rho == 0.000 on every math
#            arm measured). See PAPER_FIDELITY_AUDIT.md.
#
# COST -- READ BEFORE LAUNCHING
# -----------------------------
# One ES arm = 100 x 30 x 100 = 300,000 generations at 2048 tokens. The 07-30 full-param arm
# took **8.63 h** for exactly this shape on this model. So:
#     SEEDS=0        3 ES arms ~ 26 h  + GRPO + base ref
#     SEEDS="0 1"    3 ES arms ~ 52 h
# Default is a single seed. That is below this project's 3-seed pre-registration rule, so a
# single-seed result cannot support a per-variant claim -- it can only show whether the ES
# variants move at all in a regime where signal exists. Add seeds before concluding anything.
#
# GRPO budget: an ES arm here is 300,000 generations; GRPO at TRAIN_BS=16 x GROUP=8 = 128
# rollouts/step needs 2,343 steps to match that. BUDGET=steps (100, what the 07-30 battery used)
# gives GRPO only 12,800 generations -- a 23x disadvantage that the battery report already
# flagged as understating GRPO. Default here is `gen` so the comparison is honest; pass
# BUDGET=steps for a cheap pass and say so in the writeup.
#
# Usage:  ./run_countdown_all_arms.sh                        # single seed, generation-matched
#         SEEDS="0 1" ./run_countdown_all_arms.sh            # two seeds (~52 h of ES)
#         BUDGET=steps ./run_countdown_all_arms.sh           # cheap GRPO (understates it)
#         SMOKE=1 ./run_countdown_all_arms.sh                # 5 steps, cap 40, end-to-end check
#         STAGES="0 2" ./run_countdown_all_arms.sh           # base ref + vanilla only
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

# The countdown line is a different model from config.py's default, and every downstream path
# (results dir, GRPO checkpoint, parquet) keys off it via config.RESULTS_SUFFIX.
export MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

STAGES="${STAGES:-0 1 2 3 4 5}"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  SEEDS="${SEEDS:-0}"; STEPS="${STEPS:-5}"; N="${N:-8}"; B="${B:-16}"; EVAL_CAP="${EVAL_CAP:-40}"
  GRPO_ARGS=(BUDGET=manual STEPS=10)
else
  SEEDS="${SEEDS:-0}"; STEPS="${STEPS:-100}"; N="${N:-30}"; B="${B:-100}"; EVAL_CAP="${EVAL_CAP:-300}"
  GRPO_ARGS=(BUDGET="${BUDGET:-gen}")
fi
MAXTOK="${MAXTOK:-2048}"    # 07-30 protocol: the SAME cap for training and eval, never mixed
SIGMA="${SIGMA:-1e-3}"; ALPHA="${ALPHA:-5e-4}"   # repo reference scale = the paper's, and a=s/2

eval "$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(f'SUF={C.RESULTS_SUFFIX}')")"

OUT=axis_probe/results/countdown$SUF
# Cap goes in the path: the skip-if-exists check would otherwise hand a smoke's cap=40 baseline
# to the real cap=300 run, and every arm would be compared against a 40-question reference.
BASEOUT=axis_probe/results/countdown_base${SUF}_cap${EVAL_CAP}
LOG=axis_probe/logs/countdown$SUF
mkdir -p "$OUT" "$BASEOUT" "$LOG"
DRIVER="$LOG/driver.log"
TRAIN=axis_probe/src/es_train_axis.py

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }
say "=== countdown line start ==="
say "model=$MODEL suffix='${SUF}' seeds=[$SEEDS] N=$N B=$B steps=$STEPS maxtok=$MAXTOK smoke=$SMOKE"
say "one ES arm = $((STEPS * N * B)) generations"
say "GPU $GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$GPU" 2>/dev/null)"

want() { [[ " $STAGES " == *" $1 "* ]]; }
run_stage() {
  local id=$1 label=$2; shift 2
  want "$id" || { say "SKIP  stage $id ($label)"; return 0; }
  say "START stage $id: $label"; local t0=$SECONDS
  "$@"; local rc=$?
  say "DONE  stage $id: $label rc=$rc elapsed=$((SECONDS-t0))s"
  [[ $rc -ne 0 ]] && say "  !! stage $id FAILED -- continuing"    # a dead arm must not kill the rest
  return 0
}

stage_base() {
  [[ -f "$BASEOUT/base_summary.json" ]] && { say "  base exists, skip"; return 0; }
  $P4_PY axis_probe/eval_base.py --out_prefix "$BASEOUT/base" \
      --eval_cap "$EVAL_CAP" --max_tokens "$MAXTOK" --gpu "$GPU" --include_countdown > "$LOG/base.log" 2>&1
}
stage_grpo() { env "${GRPO_ARGS[@]}" TASK=countdown GPU="$GPU" EVAL_CAP="$EVAL_CAP" \
                   ES_GENERATIONS=$((STEPS * N * B)) \
                   TRAIN_BS=16 GROUP=8 MICRO=8 MAXRESP="$MAXTOK" MAXPROMPT=1024 \
                   ./axis_probe/run_grpo.sh; }
stage_es() {
  local variant=$1
  for seed in $SEEDS; do
    local tag="${variant}_N${N}_s${seed}"
    if [[ -f "$OUT/${tag}_summary.json" ]]; then say "  SKIP $tag (summary exists)"; continue; fi
    say "  START $tag"; local t0=$SECONDS
    $P4_PY "$TRAIN" --variant "$variant" --dataset countdown \
        --population_size "$N" --batch "$B" --num_steps "$STEPS" --max_tokens "$MAXTOK" \
        --sigma "$SIGMA" --alpha "$ALPHA" \
        --pop_seed "$seed" --gpu "$GPU" --kl --eval_final --eval_cap "$EVAL_CAP" \
        --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
    local rc=$?          # same reason: say() interpolates $(date), which would clobber $?
    say "  DONE  $tag rc=$rc elapsed=$((SECONDS-t0))s"
    [[ $rc -ne 0 ]] && say "  !! $tag FAILED -- see $LOG/$tag.log"
  done
}

run_stage 0 "untrained base reference" stage_base
run_stage 1 "GRPO"                     stage_grpo
run_stage 2 "ES vanilla"               stage_es vanilla
run_stage 3 "ES baseaxis"              stage_es baseaxis
run_stage 4 "ES momentum"              stage_es momentum

if want 5; then
  say "START stage 5: readout"
  $P4_PY - "$BASEOUT" "axis_probe/results/grpo$SUF" "$OUT" "$MODEL" <<'PY' 2>&1 | tee -a "$DRIVER"
import json, sys, glob, os
base_dir, grpo_dir, es_dir, model = sys.argv[1:5]
OOD = ["math500", "gsm8k", "svamp", "minerva_math", "olympiadbench"]

def rows_from(pattern, label=None):
    out = []
    for f in sorted(glob.glob(pattern)):
        try: d = json.load(open(f))
        except Exception: continue
        ev = d.get("eval_final") or {}
        def acc(n, k="accuracy"):
            v = ev.get(n); return v.get(k) if isinstance(v, dict) else None
        ood = [acc(s) for s in OOD]; ood = [x for x in ood if x is not None]
        seed = d.get("pop_seed"); name = label or d.get("variant") or "?"
        arm = (f"{name}/N{d.get('population_size')}x{d.get('num_steps')}/s{seed}"
               if seed is not None else
               (f"{name}/{os.path.basename(f).split('_summary')[0]}" if label == "grpo" else name))
        out.append({"arm": arm, "countdown(ID)": acc("countdown"),
                    "cd_extract": acc("countdown", "extract_rate"),
                    "OOD-avg": (sum(ood)/len(ood)) if ood else None,
                    "math500": acc("math500"),
                    "KLx1e3": (d.get("kl_proxy_drift")*1e3) if d.get("kl_proxy_drift") is not None else None,
                    "s/step": d.get("s_per_step_mean")})
    return out

rows  = rows_from(os.path.join(base_dir, "base_summary.json"), label="base")
rows += rows_from(os.path.join(grpo_dir, "*countdown*_summary.json"), label="grpo")
for v in ("vanilla", "baseaxis", "momentum"):
    rows += rows_from(os.path.join(es_dir, f"{v}_N*_summary.json"))
if not rows:
    print("no summaries yet"); sys.exit(0)

def fmt(x): return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
cols = ["arm", "countdown(ID)", "cd_extract", "math500", "OOD-avg", "KLx1e3", "s/step"]
w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print(f"\n=== {model} / countdown ===")
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))
# ID is countdown here and the math sets are OOD -- the inverse of the math line, so a drop on
# math500 is expected and is not evidence of damage.
print("\ncountdown = ID (trained on rows [300:], evaluated on [:300]); math sets = OOD.")
PY
  say "DONE  stage 5"
fi
say "=== countdown line complete ==="
