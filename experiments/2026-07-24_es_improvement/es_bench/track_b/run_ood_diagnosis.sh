#!/usr/bin/env bash
# Why are the trained arms below base OOD? Three tests, unattended (~70 min):
#   1 ES lambda grid {1,.75,.5,.25} + base, per-question -> paired McNemar + dose-response
#   2 GRPO battery per-question -> paired McNemar vs the SAME base outputs
#   3 memorization probe: base + GRPO on the GRPO train pool vs held-out
set -uo pipefail
cd "$(dirname "$0")"; TRACK_B=$(pwd); ES_BENCH=$(cd .. && pwd)
source "$ES_BENCH/track_a/env.sh"
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
HF_MERGED=${HF_MERGED:-/tmp/grpo_ckpt_grpo_full_3b_math_t2048/global_step_200/hf_merged}
THETA=$TRACK_B/results/loraes_bestshot_3b_math/theta_final.pt
D=$TRACK_B/results/ood_diag; LOGS=$TRACK_B/logs/ood_diag
mkdir -p "$D" "$LOGS"
STATUS=$D/STATUS.txt
stage(){ echo "[$(date +%F' '%T)] $*" | tee -a "$STATUS"; }
fail(){ stage "FAILED: $*"; exit 1; }

[ -f "$HF_MERGED/config.json" ] || fail "GRPO hf_merged gone from /tmp -- re-run run_grpo_chain.sh first"
[ -f "$THETA" ] || fail "ES theta_final.pt missing"

stage "=== OOD diagnosis start ==="
if [ ! -f "$D/es_eval.json" ]; then
  stage "1/4 ES lambda grid + base, perq (~45 min)"
  $P4_PY "$TRACK_B/src/eval_oldproto.py" --model "$MODEL" --theta "$THETA" \
    --rank 8 --targets q_proj,k_proj,v_proj,o_proj --lambdas 1.0,0.75,0.5,0.25 \
    --tag es --eval_cap 300 --max_tokens 2048 --perq --gpu "$GPU" \
    --out "$D/es_eval.json" > "$LOGS/es_eval.log" 2>&1 || fail "es eval, see $LOGS/es_eval.log"
fi
if [ ! -f "$D/grpo_eval.json" ]; then
  stage "2/4 GRPO battery, perq (~10 min)"
  $P4_PY "$TRACK_B/src/eval_oldproto.py" --model "$HF_MERGED" \
    --tag grpo --eval_cap 300 --max_tokens 2048 --perq --gpu "$GPU" \
    --out "$D/grpo_eval.json" > "$LOGS/grpo_eval.log" 2>&1 || fail "grpo eval"
fi
if [ ! -f "$D/overfit_grpo.json" ]; then
  stage "3/4 memorization probe (~12 min)"
  $P4_PY "$TRACK_B/src/grpo_overfit_probe.py" --model "$MODEL" \
    --out "$D/overfit_base.json" --gpu "$GPU" > "$LOGS/overfit_base.log" 2>&1 || fail "overfit base"
  $P4_PY "$TRACK_B/src/grpo_overfit_probe.py" --model "$HF_MERGED" \
    --out "$D/overfit_grpo.json" --gpu "$GPU" > "$LOGS/overfit_grpo.log" 2>&1 || fail "overfit grpo"
fi
stage "4/4 paired analysis"
$P4_PY "$TRACK_B/src/analyze_ood_diagnosis.py" --dir "$D" --out "$D/DIAGNOSIS.json" \
  > "$LOGS/analysis.log" 2>&1 || fail "analysis, see $LOGS/analysis.log"
stage "=== OOD diagnosis COMPLETE -> $D/DIAGNOSIS.json ==="
