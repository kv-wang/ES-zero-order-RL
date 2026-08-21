#!/usr/bin/env bash
# One driver for the whole cross-method comparison on ONE model, run sequentially on one GPU:
#
#   stage 0   untrained base reference   (eval only, no training)
#   stage 1   GRPO                       -> run_grpo.sh
#   stage 2   ES vanilla                 -\
#   stage 3   ES baseaxis                 |-> run_momentum.sh, variant-major, in this order
#   stage 4   ES momentum                -/
#   stage 5   combined readout across all five
#
# It DELEGATES to run_grpo.sh and run_momentum.sh rather than reimplementing them, so there is
# exactly one definition of each arm. Everything those two scripts already guarantee -- shared
# make_split, shared reward object, shared eval battery/cap/extractor, generation-count-matched
# budgets, skip-if-summary-exists resume -- is inherited unchanged.
#
# Model: whatever config.py MODEL resolves to (Qwen2.5-Math-1.5B as of 2026-08-14). Results are
# written under a per-model suffix (config.RESULTS_SUFFIX), so this can be run once per model
# without either one overwriting or silently "resuming" onto the other's output.
#
# Ordering rationale: stage 0 first because every later comparison is against it and it is the
# cheapest thing that can expose a broken model/prompt combination. GRPO before ES because it is
# the control the ES arms are argued against, and because it is the one most likely to fail on
# config (verl is strict), so a config error surfaces in ~1h rather than after ~8h of ES.
#
# Usage:  ./run_all_arms.sh                       # full run, seeds 0 1, ~10-12h
#         SEEDS=0 ./run_all_arms.sh               # single seed, ~6h
#         SMOKE=1 ./run_all_arms.sh               # 10-step ES + 20-step GRPO + 40-question eval
#         STAGES="0 2" ./run_all_arms.sh          # just base + vanilla
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

STAGES="${STAGES:-0 1 2 3 4 5}"
GPU="${GPU:-0}"
SMOKE="${SMOKE:-0}"

# SEEDS/EVAL_CAP defaults must be set INSIDE the branches. Defaulting SEEDS before this point
# would make the smoke branch's ${SEEDS:-0} a no-op, and the smoke would silently run both seeds.
if [[ "$SMOKE" == "1" ]]; then                # tiny end-to-end shakeout, not a result
  SEEDS="${SEEDS:-0}"
  EVAL_CAP="${EVAL_CAP:-40}"
  export STEPS="${STEPS:-10}" N="${N:-8}"
  GRPO_ARGS=(BUDGET=manual STEPS=20)
else
  SEEDS="${SEEDS:-0 1}"
  EVAL_CAP="${EVAL_CAP:-300}"
  GRPO_ARGS=()
fi

eval "$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(f'MODEL={C.MODEL}\nSUF={C.RESULTS_SUFFIX}')")"

BASEOUT=axis_probe/results/base$SUF
LOG=axis_probe/logs/all_arms$SUF
mkdir -p "$BASEOUT" "$LOG"
DRIVER="$LOG/driver.log"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

say "=== run_all_arms start ==="
say "model=$MODEL  results-suffix='${SUF}'  seeds=[$SEEDS]  stages=[$STAGES]  smoke=$SMOKE"
say "GPU $GPU free memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$GPU" 2>/dev/null)"

stage_wanted() { [[ " $STAGES " == *" $1 "* ]]; }

run_stage() {  # id  label  command...
  local id=$1 label=$2; shift 2
  stage_wanted "$id" || { say "SKIP  stage $id ($label) -- not in STAGES"; return 0; }
  say "START stage $id: $label"
  local t0=$SECONDS
  "$@"
  local rc=$?
  say "DONE  stage $id: $label rc=$rc elapsed=$((SECONDS-t0))s"
  # A failed stage must not kill the rest: each arm is independently useful, and a partial
  # comparison beats losing the arms that would have run after the failure.
  [[ $rc -ne 0 ]] && say "  !! stage $id FAILED -- continuing to the next stage"
  return 0
}

# ---- stage 0: untrained base reference ----
stage_base() {
  if [[ -f "$BASEOUT/base_summary.json" ]]; then
    say "  base summary exists, skipping"; return 0
  fi
  $P4_PY axis_probe/eval_base.py --out_prefix "$BASEOUT/base" \
      --eval_cap "$EVAL_CAP" --gpu "$GPU" > "$LOG/base.log" 2>&1
}

# ---- stages 1-4: delegate ----
stage_grpo() { env "${GRPO_ARGS[@]}" GPU="$GPU" EVAL_CAP="$EVAL_CAP" ./axis_probe/run_grpo.sh; }
stage_es()   { env VARIANTS="$1" SEEDS="$SEEDS" GPU="$GPU" EVAL_CAP="$EVAL_CAP" \
                   ./axis_probe/run_momentum.sh; }

run_stage 0 "untrained base reference"  stage_base
run_stage 1 "GRPO"                      stage_grpo
run_stage 2 "ES vanilla"                stage_es vanilla
run_stage 3 "ES baseaxis"               stage_es baseaxis
run_stage 4 "ES momentum"               stage_es momentum

# ---- stage 5: combined readout ----
if stage_wanted 5; then
  say "START stage 5: combined readout"
  $P4_PY - "$BASEOUT" "axis_probe/results/grpo$SUF" "axis_probe/results/momentum$SUF" "$MODEL" \
      <<'PY' 2>&1 | tee -a "$DRIVER"
import json, sys, glob, os
base_dir, grpo_dir, es_dir, model = sys.argv[1:5]
SETS = ["math500", "svamp", "gsm8k", "minerva_math", "olympiadbench", "amc23"]
OOD4 = ("svamp", "gsm8k", "minerva_math", "olympiadbench")

def rows_from(pattern, label=None):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = d.get("eval_final") or {}
        def acc(name, key="accuracy"):
            v = ev.get(name)
            return v.get(key) if isinstance(v, dict) else None
        ood = [acc(s) for s in OOD4]
        ood = [x for x in ood if x is not None]
        seed = d.get("pop_seed")
        name = label or d.get("variant") or "?"
        if seed is not None:
            # N and step count belong in the label: a 10-step N=8 smoke lands in the same
            # directory as the 200-step N=16 result and would otherwise read as one of them.
            arm = f"{name}/N{d.get('population_size')}x{d.get('num_steps')}/s{seed}"
        elif label == "grpo":
            # results/grpo holds one file per budget setting, e.g. a 400-step run next to a
            # 20-step smoke. Without the step count in the label a smoke reads as a result.
            arm = f"{name}/{os.path.basename(f)[len('grpo_'):-len('_summary.json')]}"
        else:
            arm = name
        out.append({
            "arm": arm,
            "ID(L3-5)": acc("math500", "primary_L3_5_accuracy"),
            "MATH500": acc("math500"),
            "OOD-avg": (sum(ood) / len(ood)) if len(ood) == 4 else None,
            "KLx1e3": (d.get("kl_proxy_drift") * 1e3) if d.get("kl_proxy_drift") is not None else None,
            "tie_rate": d.get("pair_tie_rate"),
            "zero_upd": d.get("zero_update_rate"),
            "s/step": d.get("s_per_step_mean"),
            "_sets": {s: acc(s) for s in SETS},
            "_ext": {s: acc(s, "extract_rate") for s in SETS},
        })
    return out

rows  = rows_from(os.path.join(base_dir, "base_summary.json"), label="base")
rows += rows_from(os.path.join(grpo_dir, "*_summary.json"),    label="grpo")
for v in ("vanilla", "baseaxis", "momentum"):
    rows += rows_from(os.path.join(es_dir, f"{v}_N*_summary.json"))

if not rows:
    print("no summaries found -- nothing completed yet"); sys.exit(0)

def fmt(x):
    return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))

print(f"\n=== {model} ===")
cols = ["arm", "ID(L3-5)", "MATH500", "OOD-avg", "KLx1e3", "tie_rate", "zero_upd", "s/step"]
w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))

print("\n=== per-dataset accuracy (extract rate in parens) ===")
w0 = max(len("arm"), *(len(r["arm"]) for r in rows))
print("arm".ljust(w0) + "  " + "  ".join(s.ljust(15) for s in SETS))
for r in rows:
    cells = []
    for s in SETS:
        a, e = r["_sets"].get(s), r["_ext"].get(s)
        cells.append("" if a is None else f"{a:.4f} ({e:.2f})" if e is not None else f"{a:.4f}")
    print(r["arm"].ljust(w0) + "  " + "  ".join(c.ljust(15) for c in cells))

# Extract rate = fraction of responses the extractor could find ANY answer in. A response it
# cannot parse scores 0 whether or not the model solved the problem, so accuracy <= extract rate
# always, and (extract_rate - accuracy) is the width of the band the true accuracy sits in.
#
# Two different numbers are in play, deliberately -- both introduced 2026-08-14, neither an
# established repo convention:
#   0.95 (here)              advisory. Flags a column worth looking at before trusting it.
#   0.80 (PREREG_BASE_MODEL) decision rule. Below it, that dataset's conclusions are void.
# Nothing calibrated either one; they are judgement calls and should be revisited if a result
# ever turns on them.
bad = [(r["arm"], s, e) for r in rows for s in SETS
       if (e := r["_ext"].get(s)) is not None and e < 0.95]
if bad:
    print("\n!! extract rate < 0.95 -- these accuracies are NOT comparable:")
    for arm, s, e in bad:
        print(f"   {arm:24s} {s:15s} {e:.3f}")
PY
  say "DONE  stage 5"
fi

say "=== run_all_arms complete ==="
