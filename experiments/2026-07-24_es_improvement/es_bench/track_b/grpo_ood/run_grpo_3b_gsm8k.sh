#!/usr/bin/env bash
# Full-param GRPO on Qwen2.5-3B-Instruct, GSM8K, 200 steps -> FSDP merge -> OOD/ID battery eval.
# Mirrors the 1.5B run in OOD_COMPARISON_REPORT.md, on the 3B model. Single GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TRACKB="$(cd "$HERE/.." && pwd)"
LOGDIR="$TRACKB/logs/exp3b_gsm8k"
RESDIR="$TRACKB/results/exp3b_gsm8k"
mkdir -p "$LOGDIR" "$RESDIR"

EXP=grpo_full_3b_gsm8k
GPU=${GPU:-0}
CKPT=/tmp/grpo_ckpt_$EXP

echo "[$(date +%T)] GRPO 3B train START ($EXP)" | tee -a "$LOGDIR/driver.log"
MODEL=Qwen/Qwen2.5-3B-Instruct GPU=$GPU STEPS=200 EXP=$EXP \
  bash "$HERE/run_grpo_train.sh" > "$LOGDIR/${EXP}_train.log" 2>&1
rc=$?
echo "[$(date +%T)] GRPO 3B train DONE rc=$rc" | tee -a "$LOGDIR/driver.log"
if [ $rc -ne 0 ]; then echo "TRAIN FAILED rc=$rc — see ${EXP}_train.log"; exit $rc; fi

ACTOR="$CKPT/global_step_200/actor"
echo "[$(date +%T)] GRPO 3B eval START (actor=$ACTOR)" | tee -a "$LOGDIR/driver.log"
PY=/home/hyin66/micromamba/envs/verl/bin/python
$PY "$HERE/eval_grpo_ood.py" --actor_dir "$ACTOR" --tag grpo_full_3b \
  --out "$RESDIR/grpo_full_3b.json" --eval_cap 300 --gpu $GPU \
  > "$LOGDIR/${EXP}_eval.log" 2>&1
rc=$?
echo "[$(date +%T)] GRPO 3B eval DONE rc=$rc -> $RESDIR/grpo_full_3b.json" | tee -a "$LOGDIR/driver.log"
exit $rc
