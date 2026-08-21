#!/usr/bin/env bash
# Adaptive-sigma experiment: vanilla ES with a FIXED sigma, versus the same arm with sigma
# scaled by the ES gradient norm. Run order: vanilla first, then the adaptive arm.
#
# The idea under test (user-proposed): a larger ES "gradient" should get a larger perturbation.
#   signal   fitness_std, which IS the ES gradient norm up to a known constant --
#            ||g_hat|| ~ sqrt(d/N) * std(f) / sigma. Unlike the probe's sum_k a_k(z+ - z-)
#            it exists for the vanilla arm, and unlike ||Delta|| (pinned by z-scoring to the
#            constant alpha*sqrt(d/N)) it actually varies: measured CV 0.59-0.64.
#   law      sigma_t = sigma0 * clip( (fs_ema / fs_ref)^gain , 1/clip , clip )
#   causal   sigma_t uses only fitness_std from steps < t.
#   alpha    left untouched, as requested.
#
# TWO THINGS TO WATCH, both pre-registered here so the readout is not post-hoc:
#
# 1. sigma does NOT change the step size, so radius is already isolated. The coefficients
#    are (alpha/N)*z and the commit adds raw eps, not sigma*eps, so ||Delta|| ~ alpha*sqrt(d/N)
#    regardless of sigma. Deeper reason: in the linear regime f_i - f_bar ~ sigma<g,eps_i> and
#    std(f) ~ sigma*||g||, so z_i ~ <g,eps_i>/||g|| and sigma cancels exactly. Changing sigma
#    has NO first-order effect on the update. Its one dominant second-order path here is reward
#    QUANTIZATION: at B=8 the reward takes 9 values and 64-75% of probe pairs tie, so a larger
#    sigma pushes members into distinct buckets and lowers the tie rate. Read the TIE RATE
#    before accuracy -- if ties do not move, accuracy cannot, and its null carries no signal.
#    (This corrects the first version of this header, written before the derivation.)
#
# 2. The control law may be self-reinforcing in the wrong direction. On this task 52-65% of
#    steps are ties and ~17% are zero-update (fitness_std = 0). Zero signal pushes sigma DOWN,
#    which makes ties more likely, which pushes sigma down again. --sigma_adapt_clip bounds
#    the excursion (default sigma0/2 .. 2*sigma0) and the EMA smooths single-step zeros, but
#    if sigma_ratio sits at the lower clip for most steps that is the failure mode, not a
#    tuning problem. `sigma_ratio_min/mean/max` are in the summary for exactly this check.
#
# Settings otherwise identical to the 2026-08-10 standard run: Qwen2.5-Math-1.5B, MATH L3-5,
# B=8, sigma0=1e-3, alpha=5e-4, fp16, CRN, N=16, 200 steps, seeds {0,1},
# --kl --eval_final --eval_cap 300. Extractor is the post-fix brace-counting version.
#
# WALL CLOCK: ~1.23h per arm-seed. Default (2 arms x 2 seeds) is ~4.9h, sequential on GPU 0.
#
#   Usage:  ./run_sigma_adapt.sh                    # seeds 0 1, both arms (~4.9h)
#           SEEDS=0 ./run_sigma_adapt.sh            # seed 0 only (~2.5h)
#           N=8 STEPS=20 SEEDS=0 ./run_sigma_adapt.sh   # smoke
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

SEEDS="${SEEDS:-0 1}"
N="${N:-16}"
STEPS="${STEPS:-200}"
GPU="${GPU:-0}"
EVAL_CAP="${EVAL_CAP:-300}"
GAIN="${GAIN:-1.0}"
CLIP="${CLIP:-2.0}"
EMA="${EMA:-0.9}"
AWARM="${AWARM:-10}"

# Results are scoped by model: config.RESULTS_SUFFIX is '' for the historical -Instruct
# runs and '_base' for Qwen2.5-Math-1.5B. Without this, the SKIP-if-summary-exists check
# below would treat the other model's completed arms as this model's and run nothing.
SUF=$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.RESULTS_SUFFIX)")
OUT=axis_probe/results/sigma_adapt$SUF
LOG=axis_probe/logs/sigma_adapt$SUF
mkdir -p "$OUT" "$LOG"
TRAIN=axis_probe/src/es_train_axis.py
DRIVER="$LOG/driver.log"

echo "[$(date +%F_%T)] start: arms=[vanilla vanilla_sigadapt] seeds=[$SEEDS] N=$N steps=$STEPS gain=$GAIN clip=$CLIP ema=$EMA warmup=$AWARM" | tee "$DRIVER"

run_arm() {  # tag  extra-args...
  local tag=$1; shift
  if [[ -f "$OUT/${tag}_summary.json" ]]; then
    echo "[$(date +%F_%T)] SKIP  $tag (summary exists)" | tee -a "$DRIVER"
    return 0
  fi
  echo "[$(date +%F_%T)] START $tag" | tee -a "$DRIVER"
  local t0=$SECONDS
  $P4_PY "$TRAIN" --variant vanilla --population_size "$N" --num_steps "$STEPS" \
      --gpu "$GPU" --kl --eval_final --eval_cap "$EVAL_CAP" \
      --out_prefix "$OUT/$tag" "$@" > "$LOG/$tag.log" 2>&1
  local rc=$?
  echo "[$(date +%F_%T)] DONE  $tag rc=$rc elapsed=$((SECONDS-t0))s" | tee -a "$DRIVER"
  [[ $rc -ne 0 ]] && echo "  !! failed, see $LOG/$tag.log" | tee -a "$DRIVER"
  return 0
}

for seed in $SEEDS; do                      # fixed-sigma baseline first, then adaptive
  run_arm "vanilla_N${N}_s${seed}" --pop_seed "$seed"
done
echo "[$(date +%F_%T)] --- vanilla (fixed sigma) complete ---" | tee -a "$DRIVER"

for seed in $SEEDS; do
  run_arm "sigadapt_N${N}_s${seed}" --pop_seed "$seed" \
      --sigma_adapt --sigma_adapt_gain "$GAIN" --sigma_adapt_clip "$CLIP" \
      --sigma_adapt_ema "$EMA" --sigma_adapt_warmup "$AWARM"
done
echo "[$(date +%F_%T)] --- sigma-adaptive complete ---" | tee -a "$DRIVER"
echo "[$(date +%F_%T)] ALL ARMS COMPLETE" | tee -a "$DRIVER"

$P4_PY - "$OUT" <<'PY' | tee -a "$DRIVER"
import json, sys, glob, os
out = sys.argv[1]
SETS = ["math500", "svamp", "gsm8k", "minerva_math", "olympiadbench", "amc23"]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*_summary.json"))):
    d = json.load(open(f))
    ev = d.get("eval_final") or {}
    def acc(n, k="accuracy"):
        v = ev.get(n)
        return v.get(k) if isinstance(v, dict) else None
    ood = [acc(s) for s in ("svamp", "gsm8k", "minerva_math", "olympiadbench")]
    ood = [x for x in ood if x is not None]
    rows.append({
        "arm": os.path.basename(f).replace("_summary.json", "").replace("_N16", ""),
        "ID(L3-5)": acc("math500", "primary_L3_5_accuracy"),
        "OOD-avg": (sum(ood) / len(ood)) if len(ood) == 4 else None,
        "KLx1e3": (d["kl_proxy_drift"] * 1e3) if d.get("kl_proxy_drift") is not None else None,
        "zero_upd": d.get("zero_update_rate"),
        "sig_ratio(min/mean/max)": (None if not d.get("sigma_adapt") else
            f"{d['sigma_ratio_min']:.2f}/{d['sigma_ratio_mean']:.2f}/{d['sigma_ratio_max']:.2f}"),
        "s/step": d.get("s_per_step_mean"),
    })
if not rows:
    print("no summaries found"); sys.exit(0)
def fmt(x):
    return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
cols = list(rows[0].keys())
w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print("\n=== fixed sigma vs gradient-scaled sigma ===")
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))
print("\nRead sig_ratio first. Pinned at the lower clip => the feedback loop ran the wrong way")
print("(zero signal shrinks sigma, which makes ties more likely). Pinned at the upper clip =>")
print("the gain is too high for the clip. Only a ratio that moves inside the band is a real test.")
print("\nCaveats: alpha is fixed but effective_lr = alpha*sigma_t is not, so radius and step size")
print("are confounded; 2 seeds; absolute accuracies now use the fixed extractor and are NOT")
print("comparable to anything produced before 2026-08-10.")
PY
