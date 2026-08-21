#!/usr/bin/env bash
# Qwen2.5-3B base model on the MATH L3-5 math-reasoning harness.
#
# Qwen2.5-Math series has no 3B checkpoint; default MODEL is Qwen/Qwen2.5-3B (base).
# data_math.build_prompt auto-selects 4-shot for non-Instruct eos tokens.
#
# Fixed run order (unlike run_all_arms.sh, which runs GRPO before ES):
#   1  untrained base reference eval
#   2  vanilla ES train + eval
#   3  GRPO train + eval
#   4  combined readout (base / vanilla / grpo)
#
# Delegates to eval_base.py, run_momentum.sh, and run_grpo.sh so reward, data split,
# eval battery, resume/skip behavior, and generation-matched GRPO budget stay single-sourced.
#
# Usage:
#   ./axis_probe/run_qwen25_3b_math.sh
#   SEEDS=0 ./axis_probe/run_qwen25_3b_math.sh
#   SMOKE=1 ./axis_probe/run_qwen25_3b_math.sh
set -uo pipefail

cd "$(dirname "$0")/.."                       # es_bench dir
source axis_probe/env.sh

export MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
export TASK="${TASK:-math}"
export GPU="${GPU:-0}"
export SEEDS="${SEEDS:-0}"
export N="${N:-16}"
export STEPS="${STEPS:-200}"
export EVAL_CAP="${EVAL_CAP:-300}"
export SMOKE="${SMOKE:-0}"

export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"
export ROLLOUT_MEM="${ROLLOUT_MEM:-0.50}"
export MICRO="${MICRO:-4}"

if [[ "$SMOKE" == "1" ]]; then
  SEEDS="${SEEDS:-0}"
  EVAL_CAP="${EVAL_CAP:-40}"
  export STEPS="${STEPS:-10}" N="${N:-8}"
  GRPO_ARGS=(BUDGET=manual STEPS=20)
else
  GRPO_ARGS=()
fi

export ES_BATCH="${ES_BATCH:-$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config;print(config.BATCH_SIZE)")}"
export ES_GENERATIONS="${ES_GENERATIONS:-$((STEPS * N * ES_BATCH))}"

eval "$($P4_PY -c "import sys;sys.path.insert(0,'axis_probe/src');import config as C;print(f'SUF={C.RESULTS_SUFFIX}')")"

BASEOUT=axis_probe/results/base$SUF
GRPOOUT=axis_probe/results/grpo$SUF
ESOUT=axis_probe/results/momentum$SUF
LOG=axis_probe/logs/qwen25_3b_math$SUF
mkdir -p "$BASEOUT" "$GRPOOUT" "$ESOUT" "$LOG"
DRIVER="$LOG/driver.log"

say() { echo "[$(date +%F_%T)] $*" | tee -a "$DRIVER"; }

run_step() {
  local label=$1; shift
  say "START $label"
  local t0=$SECONDS
  "$@"
  local rc=$?
  say "DONE  $label rc=$rc elapsed=$((SECONDS-t0))s"
  [[ $rc -ne 0 ]] && say "  !! $label FAILED -- continuing"
  return 0
}

say "=== qwen25-3b math (base) start ==="
say "MODEL=$MODEL SUF=$SUF SEEDS=[$SEEDS] N=$N STEPS=$STEPS ES_BATCH=$ES_BATCH"
say "ES_GENERATIONS=$ES_GENERATIONS GPU=$GPU EVAL_CAP=$EVAL_CAP SMOKE=$SMOKE"
say "GPU_MEM_UTIL=$GPU_MEM_UTIL ROLLOUT_MEM=$ROLLOUT_MEM MICRO=$MICRO"
say "GPU $GPU free memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$GPU" 2>/dev/null)"

# ---- 1. base reference eval ----
run_step "base reference eval" bash -c "
  if [[ -f '$BASEOUT/base_summary.json' ]]; then
    echo '  base summary exists, skipping'; exit 0
  fi
  $P4_PY axis_probe/eval_base.py --out_prefix '$BASEOUT/base' \
      --eval_cap '$EVAL_CAP' --gpu '$GPU' > '$LOG/base.log' 2>&1
"

# ---- 2. vanilla ES ----
run_step "vanilla ES" env VARIANTS=vanilla SEEDS="$SEEDS" GPU="$GPU" EVAL_CAP="$EVAL_CAP" \
  GPU_MEM_UTIL="$GPU_MEM_UTIL" STEPS="$STEPS" N="$N" \
  ./axis_probe/run_momentum.sh

# ---- 3. GRPO ----
run_step "GRPO" env "${GRPO_ARGS[@]}" GPU="$GPU" EVAL_CAP="$EVAL_CAP" TASK="$TASK" MODEL="$MODEL" \
  ./axis_probe/run_grpo.sh

# ---- 4. readout ----
say "START combined readout"
$P4_PY - "$BASEOUT" "$GRPOOUT" "$ESOUT" "$MODEL" <<'PY' 2>&1 | tee -a "$DRIVER"
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
            arm = f"{name}/N{d.get('population_size')}x{d.get('num_steps')}/s{seed}"
        elif label == "grpo":
            arm = f"{name}/{os.path.basename(f)[len('grpo_'):-len('_summary.json')]}"
        else:
            arm = name
        out.append({
            "arm": arm,
            "ID(L3-5)": acc("math500", "primary_L3_5_accuracy"),
            "MATH500": acc("math500"),
            "OOD-avg": (sum(ood) / len(ood)) if len(ood) == 4 else None,
            "KLx1e3": (d.get("kl_proxy_drift") * 1e3) if d.get("kl_proxy_drift") is not None else None,
        })
    return out

rows  = rows_from(os.path.join(base_dir, "base_summary.json"), label="base")
rows += rows_from(os.path.join(es_dir, "vanilla_N*_summary.json"))
rows += rows_from(os.path.join(grpo_dir, "*_summary.json"), label="grpo")

if not rows:
    print("no summaries found -- nothing completed yet"); sys.exit(0)

def fmt(x):
    return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))

print(f"\n=== {model} (base / vanilla ES / GRPO) ===")
cols = ["arm", "ID(L3-5)", "MATH500", "OOD-avg", "KLx1e3"]
w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(fmt(r.get(c)).ljust(w[c]) for c in cols))
PY
say "DONE  combined readout"

say "=== qwen25-3b math (base) complete ==="
