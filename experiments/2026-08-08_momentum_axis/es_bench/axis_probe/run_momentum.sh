#!/usr/bin/env bash
# A1 momentum-axis training + eval, then base-axis and vanilla reproduced at the SAME settings.
#
# Order is variant-major: all momentum seeds first, then all baseaxis seeds, then all vanilla.
# That way the new arm is complete first, and a killed job still leaves a whole momentum arm.
#
# Settings inherited verbatim from base_axis_probe/run_pilot.sh and config.py:
#   Qwen2.5-Math-1.5B, MATH L3-5, B=8, sigma=1e-3, alpha=5e-4, fp16, CRN batches,
#   N=16, 200 steps, a_max=1.0, seeds {0,1}, --kl --eval_final --eval_cap 300.
# Momentum-only additions: --mom_beta 0.9, --mom_warmup 1.
#
# Why the baseaxis and vanilla arms are re-run rather than reused from 2026-07-23: re-running
# puts all three arms in one results directory under one harness, and gives a reproducibility
# check -- the worker's baseaxis path is proven bit-identical to the frozen 07-23 worker
# (axis_probe/tests/test_axis_worker.py), the trainer consumes the prng in the same order, and
# the pilot used these same two seeds. Matching rows are evidence the protocol is deterministic;
# diverging rows would point at vLLM generation nondeterminism and are worth knowing about.
#
# mom_warmup=1 is what makes the arms comparable: measured on all five 07-23 runs,
# ||theta0-theta_t|| crosses anchor_threshold=1.0 at step 1 and stays above it, so the baseaxis
# probe is active 199/200 steps. Gating momentum at step 1 puts both probe arms on an identical
# prng stream -- same radii, same 12 Gaussian seeds, every step.
#
# WALL CLOCK: measured from the eight 2026-07-23 summaries, one arm-seed is 1.21h
# (1.20h training at ~21.5 s/step x 200, plus ~0.8min eval). The default (3 arms x 2 seeds)
# is ~7.3h. Only GPU 0 is available on this box, so everything runs sequentially.
#
#   Usage:  ./run_momentum.sh                  # seeds 0 1, all three arms (~7.3h)
#           SEEDS=0 ./run_momentum.sh          # seed 0 only (~3.6h)
#           VARIANTS=momentum ./run_momentum.sh    # just the new arm (~2.4h)
#           N=8 STEPS=20 SEEDS=0 ./run_momentum.sh # smoke
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

SEEDS="${SEEDS:-0 1}"                          # pilot seeds
VARIANTS="${VARIANTS:-momentum baseaxis vanilla}"   # requested order
N="${N:-16}"
STEPS="${STEPS:-200}"
A_MAX="${A_MAX:-1.0}"
ANCHOR_THRESHOLD="${ANCHOR_THRESHOLD:-1.0}"
MOM_BETA="${MOM_BETA:-0.9}"
MOM_WARMUP="${MOM_WARMUP:-1}"
GPU="${GPU:-0}"
EVAL_CAP="${EVAL_CAP:-300}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

# Results are scoped by model: config.RESULTS_SUFFIX is '' for the historical -Instruct
# runs and '_base' for Qwen2.5-Math-1.5B. Without this, the SKIP-if-summary-exists check
# below would treat the other model's completed arms as this model's and run nothing.
SUF=$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.RESULTS_SUFFIX)")
OUT=axis_probe/results/momentum$SUF
LOG=axis_probe/logs/momentum$SUF
mkdir -p "$OUT" "$LOG"
TRAIN=axis_probe/src/es_train_axis.py
DRIVER="$LOG/driver.log"

echo "[$(date +%F_%T)] start: variants=[$VARIANTS] seeds=[$SEEDS] N=$N steps=$STEPS a_max=$A_MAX beta=$MOM_BETA warmup=$MOM_WARMUP gpu_mem_util=$GPU_MEM_UTIL" | tee "$DRIVER"

run_arm() {  # variant seed
  local variant=$1 seed=$2
  local tag="${variant}_N${N}_s${seed}"
  if [[ -f "$OUT/${tag}_summary.json" ]]; then
    echo "[$(date +%F_%T)] SKIP  $tag (summary exists)" | tee -a "$DRIVER"
    return 0
  fi
  echo "[$(date +%F_%T)] START $tag" | tee -a "$DRIVER"
  local t0=$SECONDS
  $P4_PY "$TRAIN" \
      --variant "$variant" --population_size "$N" --num_steps "$STEPS" \
      --pop_seed "$seed" --gpu "$GPU" --a_max "$A_MAX" \
      --anchor_threshold "$ANCHOR_THRESHOLD" \
      --mom_beta "$MOM_BETA" --mom_warmup "$MOM_WARMUP" \
      --gpu_mem_util "$GPU_MEM_UTIL" \
      --kl --eval_final --eval_cap "$EVAL_CAP" \
      --out_prefix "$OUT/$tag" > "$LOG/$tag.log" 2>&1
  local rc=$?
  echo "[$(date +%F_%T)] DONE  $tag rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$DRIVER"
  [[ $rc -ne 0 ]] && echo "  !! failed, see $LOG/$tag.log" | tee -a "$DRIVER"
  return 0                                   # keep going; a dead arm must not kill the rest
}

for variant in $VARIANTS; do                 # variant-major: finish momentum before base/vanilla
  for seed in $SEEDS; do
    run_arm "$variant" "$seed"
  done
  echo "[$(date +%F_%T)] --- $variant arm complete ---" | tee -a "$DRIVER"
done
echo "[$(date +%F_%T)] ALL ARMS COMPLETE" | tee -a "$DRIVER"

# ---- readout: the three new arms, plus the frozen 07-23 rows for reference ----
OLD_RESULTS=../../2026-07-23_lora/es_bench/base_axis_probe/results   # cwd is <try>/es_bench
$P4_PY - "$OUT" "$OLD_RESULTS" <<'PY' | tee -a "$DRIVER"
import json, sys, glob, os
new_dir, old_dir = sys.argv[1], sys.argv[2]

SETS = ["math500", "svamp", "gsm8k", "minerva_math", "olympiadbench", "amc23"]

def collect(pattern, suffix=""):
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
        ood = [acc(s) for s in ("svamp", "gsm8k", "minerva_math", "olympiadbench")]
        ood = [x for x in ood if x is not None]
        out.append({
            "arm": f"{d.get('variant')}{suffix}/s{d.get('pop_seed')}",
            "ID(L3-5)": acc("math500", "primary_L3_5_accuracy"),
            "OOD-avg": (sum(ood) / len(ood)) if len(ood) == 4 else None,
            "KLx1e3": (d.get("kl_proxy_drift") * 1e3) if d.get("kl_proxy_drift") is not None else None,
            "cum_disp": d.get("final_cum_disp"),
            "cum_frac": d.get("final_cum_disp_frac"),
            "tie_rate": d.get("pair_tie_rate"),
            "s/step": d.get("s_per_step_mean"),
            "_per_set": {s: (acc(s), acc(s, "extract_rate")) for s in SETS},
        })
    return out

rows = []
for v in ("momentum", "baseaxis", "vanilla"):
    rows += collect(os.path.join(new_dir, f"{v}_N*_summary.json"))
for v in ("baseaxis", "vanilla"):
    for sub in ("pilot", "confirm"):
        rows += collect(os.path.join(old_dir, sub, f"{v}_N*_summary.json"), suffix="*")
if not rows:
    print("no summaries found"); sys.exit(0)

def fmt(x):
    return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
cols = [c for c in rows[0].keys() if not c.startswith("_")]
w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print("\n=== A1 momentum vs base-axis vs vanilla (same settings) ===")
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))

# Per-dataset, because OOD-avg is an unweighted mean over sets whose base rates differ ~5x
# (svamp ~0.92 vs minerva ~0.17) and so corresponds to no meaningful quantity. Each cell is
# accuracy/extract-rate: on minerva and olympiadbench a third of responses yield no parseable
# answer, so an accuracy move there can be an extraction move.
print("\n=== per-dataset  (accuracy / extract-rate) ===")
hdr = ["arm"] + SETS
cells = {r["arm"]: {s: ("" if r["_per_set"][s][0] is None
                        else f"{r['_per_set'][s][0]:.3f}/{r['_per_set'][s][1]:.2f}")
                    for s in SETS} for r in rows}
w2 = {"arm": max(len("arm"), *(len(r["arm"]) for r in rows))}
for s in SETS:
    w2[s] = max(len(s), *(len(cells[r["arm"]][s]) for r in rows))
print("  ".join(h.ljust(w2[h]) for h in hdr))
for r in rows:
    print("  ".join([r["arm"].ljust(w2["arm"])] + [cells[r["arm"]][s].ljust(w2[s]) for s in SETS]))

print("\n* = 2026-07-23 run, shown for reference only. Rows without * were produced by this script.")
print("Reproducibility check: baseaxis/s0 and baseaxis*/s0 share settings, seed and worker logic,")
print("so they should agree; a gap is vLLM generation nondeterminism, not a config difference.")
print("cum_disp sign: momentum + = along the recent trajectory; baseaxis + = toward theta0.")
print("tie_rate and cum_frac are new fields, blank on the 07-23 rows.")
print("\nCaveats that survive this run:")
print("  - momentum and baseaxis differ in TWO ways, the axis and its radius policy; one run")
print("    cannot separate them (see PREREG).")
print("  - the pre-registered N=12 vanilla control still does not exist, so probe-vs-vanilla")
print("    cannot separate 'mechanism works' from 'four fewer Gaussian explorers'.")
print("  - absolute accuracies are depressed by the unfixed \\boxed{} extractor bug. It is")
print("    common-mode across arms, so paired comparisons hold and absolute numbers do not.")
PY
