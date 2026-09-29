#!/usr/bin/env bash
# Phase D -- build eval spec from whatever trained, evaluate, then assemble report.md + results.csv.
# Always writes the report, even if the eval is cut by its deadline or an arm is missing.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
OUT=track_b/results/overnight; LOG=track_b/logs/overnight
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/phaseD.driver.log"; }

log "=== Phase D START ==="
$P4_PY track_b/src/build_eval_spec.py 2>&1 | tee -a "$LOG/phaseD.driver.log"
if [ -s "$OUT/eval_spec.json" ]; then
  log "running battery (deadline $(date -d @${DL_EVAL:-0} +%T 2>/dev/null || echo none))"
  $P4_PY track_b/src/overnight_eval.py --model "$MODEL" --spec "$OUT/eval_spec.json" \
    --out "$OUT/eval_results.json" --max_tokens 512 --kl_probe_n 64 \
    --deadline_ts "${DL_EVAL:-0}" --gpu $GPU > "$LOG/phaseD_eval.log" 2>&1 \
    && log "eval DONE" || log "eval FAIL rc=$? (report will state what is missing)"
else
  log "no eval spec -- skipping battery"
fi
log "assembling report"
$P4_PY track_b/src/make_overnight_report.py 2>&1 | tee -a "$LOG/phaseD.driver.log"
log "=== Phase D DONE ==="
