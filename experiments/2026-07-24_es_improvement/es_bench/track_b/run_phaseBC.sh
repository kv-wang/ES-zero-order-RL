#!/usr/bin/env bash
# Phase B (LoRA-ES main arms) + Phase C (baselines), unattended with auto-gate relaunch.
#
# HARDWARE REALITY: only GPU0 is present on this node tonight (nvidia-smi shows a single H200),
# so the spec's "GPU0: GSM8K / GPU1: MATH" parallel plan is run SEQUENTIALLY. Each arm gets a
# wall-clock deadline instead of a step count, so the night ends with evaluated checkpoints
# rather than an unfinished run; the achieved step count is recorded in each summary.
#
# Auto-gates are enforced inside es_lora_main.py, which exits 90 (step-60 signal gate) or
# 91 (KL guard). Each task gets at most ONE relaunch, per spec.
set -uo pipefail
cd "$(dirname "$0")/.."
source track_a/env.sh
GPU=${GPU:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
N=${N:-30}
B=${B:-200}
STEPS=${STEPS:-300}
SEED=${SEED:-42}
OUT=track_b/results/overnight; LOG=track_b/logs/overnight
mkdir -p "$OUT" "$LOG"
log(){ echo "[$(date +%T)] $*" | tee -a "$LOG/phaseBC.driver.log"; }

SS=$OUT/phaseA/SIGMA_STAR.json
if [ ! -f "$SS" ]; then log "FATAL: no $SS (Phase A failed)"; exit 1; fi
SIGMA=$($P4_PY -c "import json;print(json.load(open('$SS'))['sigma_star'])")
ALPHA=$($P4_PY -c "import json;print(json.load(open('$SS'))['alpha'])")
GRID=$($P4_PY -c "import json;print(','.join(str(c['sigma']) for c in json.load(open('$SS'))['lora_curve']))")
log "Phase A selected sigma*=$SIGMA alpha=$ALPHA (grid: $GRID)"

next_sigma(){ $P4_PY -c "
g=sorted(float(x) for x in '$GRID'.split(','))
s=float('$1')
nxt=[x for x in g if x>s*1.001]
print(nxt[0] if nxt else -1)"; }

# run_arm <dataset> <deadline_ts>
run_arm(){
  local DS=$1
  local DL=$2
  local sig=$SIGMA
  local alp=$ALPHA
  local tag="loraes_${DS}"
  local tries=0
  while :; do
    log "$tag start (sigma=$sig alpha=$alp deadline=$(date -d @$DL +%T))"
    $P4_PY track_b/src/es_lora_main.py --model "$MODEL" --dataset "$DS" \
      --population_size $N --batch $B --num_steps $STEPS \
      --sigma "$sig" --alpha "$alp" --pop_seed $SEED --data_seed $SEED \
      --max_tokens 512 --chunk ${CHUNK:-6000} --ckpt_every 25 --probe_every 25 \
      --deadline_ts "$DL" --gpu $GPU \
      --out_prefix "$OUT/$tag" --ckpt_dir "$OUT/ckpt_$tag" \
      > "$LOG/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then log "$tag DONE"; return 0; fi
    if [ $tries -ge 1 ]; then log "$tag GATE rc=$rc but relaunch budget spent -- moving on"; return $rc; fi
    tries=1
    if [ $rc -eq 90 ]; then
      local ns=$(next_sigma "$sig")
      if [ "$ns" = "-1" ]; then
        alp=$sig; log "$tag rc=90 at sigma grid max -> retry with alpha=sigma=$alp"
      else
        sig=$ns; alp=$($P4_PY -c "print($ns/2)"); log "$tag rc=90 -> retry with next sigma=$sig alpha=$alp"
      fi
      mv "$OUT/${tag}.jsonl" "$OUT/${tag}_gate90_attempt1.jsonl" 2>/dev/null
      mv "$OUT/${tag}_summary.json" "$OUT/${tag}_gate90_attempt1_summary.json" 2>/dev/null
    elif [ $rc -eq 91 ]; then
      alp=$($P4_PY -c "print($alp/2)"); log "$tag rc=91 KL guard -> retry with alpha=$alp"
      mv "$OUT/${tag}.jsonl" "$OUT/${tag}_gate91_attempt1.jsonl" 2>/dev/null
      mv "$OUT/${tag}_summary.json" "$OUT/${tag}_gate91_attempt1_summary.json" 2>/dev/null
    else
      log "$tag CRASHED rc=$rc -- see $LOG/${tag}.log; not retrying a crash, moving on"; return $rc
    fi
  done
}

log "=== Phase B START (model=$MODEL N=$N B=$B seed=$SEED) ==="
run_arm gsm8k "${DL_GSM8K:-0}"
run_arm math  "${DL_MATH:-0}"
log "=== Phase B DONE ==="

# ---- Phase C.2: full-param ES iso-B timing arm (primary purpose = s/step + VRAM vs LoRA) ----
log "=== Phase C: full-param ES iso-B (GSM8K, N=$N B=$B) ==="
mkdir -p /tmp/overnight_ckpt_fulles
$P4_PY track_b/src/es_train_fullparam.py --reward binary --model "$MODEL" --dataset gsm8k \
  --population_size $N --batch $B --num_steps ${FULLES_STEPS:-60} --sigma 1e-3 --alpha 5e-4 \
  --pop_seed $SEED --gpu $GPU --ckpt_dir /tmp/overnight_ckpt_fulles \
  --out_prefix "$OUT/fulles_isoB_gsm8k" > "$LOG/fulles_isoB.log" 2>&1 \
  && log "full-param iso-B DONE" || log "full-param iso-B FAIL rc=$?"

log "=== Phase B+C ALL DONE ==="
