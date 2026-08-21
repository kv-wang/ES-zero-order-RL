#!/usr/bin/env bash
# Three-arm axis-probe comparison, run SEQUENTIALLY in this order: momentum -> baseaxis -> vanilla.
#
# Config is the frozen 2026-07-23 pilot config (config.py: Qwen2.5-Math-1.5B, MATH L3-5, B=8,
# sigma=1e-3, alpha=5e-4, 200 steps, fp16) with N=16 and a_max=1.0, so the baseaxis and vanilla
# arms here are directly comparable to base_axis_probe/results/{pilot,confirm}.
#
# Each arm trains, then evaluates the final weights on the 6-set battery (cap 300) and records a
# KL-to-base proxy. Arms are skipped if their *_summary.json already exists, so a killed run can
# be resumed by re-invoking this script.
#
#   Usage:  ./run_three_arms.sh                  # seed 0, ~7h on one H200
#           SEEDS="0 1 2" ./run_three_arms.sh    # 3 seeds, ~21h
#           N=8 STEPS=20 SEEDS=0 ./run_three_arms.sh   # smoke
#
# Only one GPU is visible on this box, so there is no lane parallelism; the ordering is the
# user-requested A1 -> base -> vanilla.
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

SEEDS="${SEEDS:-0}"
N="${N:-16}"
STEPS="${STEPS:-200}"
A_MAX="${A_MAX:-1.0}"
MOM_BETA="${MOM_BETA:-0.9}"
MOM_WARMUP="${MOM_WARMUP:-5}"
GPU="${GPU:-0}"
EVAL_CAP="${EVAL_CAP:-300}"

# Results are scoped by model: config.RESULTS_SUFFIX is '' for the historical -Instruct
# runs and '_base' for Qwen2.5-Math-1.5B. Without this, the SKIP-if-summary-exists check
# below would treat the other model's completed arms as this model's and run nothing.
SUF=$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.RESULTS_SUFFIX)")
OUT=axis_probe/results/three_arms$SUF
LOG=axis_probe/logs/three_arms$SUF
mkdir -p "$OUT" "$LOG"
TRAIN=axis_probe/src/es_train_axis.py
DRIVER="$LOG/driver.log"

run_arm() {  # variant seed
  local variant=$1 seed=$2
  local tag="${variant}_N${N}_s${seed}"
  if [[ -f "$OUT/${tag}_summary.json" ]]; then
    echo "[$(date +%F_%T)] SKIP $tag (summary exists)" | tee -a "$DRIVER"
    return 0
  fi
  echo "[$(date +%F_%T)] START $tag" | tee -a "$DRIVER"
  local t0=$SECONDS
  $P4_PY "$TRAIN" \
      --variant "$variant" --population_size "$N" --num_steps "$STEPS" \
      --pop_seed "$seed" --gpu "$GPU" --a_max "$A_MAX" \
      --mom_beta "$MOM_BETA" --mom_warmup "$MOM_WARMUP" \
      --kl --eval_final --eval_cap "$EVAL_CAP" \
      --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  local rc=$?
  echo "[$(date +%F_%T)] DONE  $tag rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$DRIVER"
  [[ $rc -ne 0 ]] && echo "  !! see $LOG/$tag.log" | tee -a "$DRIVER"
  return $rc
}

echo "[$(date +%F_%T)] three-arm run start: N=$N steps=$STEPS seeds=[$SEEDS] a_max=$A_MAX" > "$DRIVER"
for seed in $SEEDS; do
  for variant in momentum baseaxis vanilla; do      # requested order: A1 -> base -> vanilla
    run_arm "$variant" "$seed"
  done
done
echo "[$(date +%F_%T)] ALL ARMS COMPLETE" | tee -a "$DRIVER"

# ---- compact side-by-side readout ----
$P4_PY - "$OUT" <<'PY' | tee -a "$DRIVER"
import json, sys, glob, os
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*_summary.json"))):
    d = json.load(open(f))
    ev = d.get("eval_final", {})
    def acc(name, key="accuracy"):
        v = ev.get(name)
        return v.get(key) if isinstance(v, dict) else None
    rows.append({
        "arm": f"{d['variant']}/s{d['pop_seed']}",
        "ID(L3-5)": acc("math500", "primary_L3_5_accuracy"),
        "svamp": acc("svamp"), "gsm8k": acc("gsm8k"),
        "minerva": acc("minerva_math"), "olympiad": acc("olympiadbench"),
        "KL": d.get("kl_proxy_drift"),
        "cum_disp": d.get("final_cum_disp"),
        "cum_frac": d.get("final_cum_disp_frac"),
        "tie_rate": d.get("pair_tie_rate"),
        "probe_steps": d.get("steps_probe_active"),
        "s/step": d.get("s_per_step_mean"),
    })
if not rows:
    print("no summaries found"); sys.exit(0)
cols = list(rows[0].keys())
w = {c: max(len(c), *(len(("" if r[c] is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])))) for r in rows)) for c in cols}
print("\n=== three-arm readout ===")
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(("" if r[c] is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]))).ljust(w[c]) for c in cols))
print("\nsign semantics: cum_disp is + toward theta0 for baseaxis, + along recent trajectory for momentum;")
print("vanilla has no axis so its disp fields are 0 by construction.")
PY
