#!/usr/bin/env bash
# Wait for the GRPO 3B run to finish (single GPU), then run LoRA-ES on 3B GSM8K.
set -uo pipefail
cd "$(dirname "$0")/.."                 # es_bench
source track_a/env.sh
LOGDIR=track_b/logs/exp3b_gsm8k
RESDIR=track_b/results/exp3b_gsm8k
mkdir -p "$LOGDIR" "$RESDIR"

echo "[$(date +%T)] watcher: waiting for GRPO 3B to finish..." | tee -a "$LOGDIR/driver.log"
# GRPO done == its eval json exists, OR driver.log logged eval DONE (covers train-fail case)
for i in $(seq 1 240); do   # up to 4h
  if [ -f "$RESDIR/grpo_full_3b.json" ] || grep -q "GRPO 3B eval DONE" "$LOGDIR/driver.log" 2>/dev/null; then
    break
  fi
  # bail if GRPO process died without producing output
  if ! pgrep -f "main_ppo.*grpo_full_3b_gsm8k|run_grpo_3b_gsm8k" >/dev/null 2>&1 && ! grep -q "GRPO 3B eval START" "$LOGDIR/driver.log" 2>/dev/null; then
    echo "[$(date +%T)] watcher: GRPO process gone before eval — check train log" | tee -a "$LOGDIR/driver.log"; break
  fi
  sleep 60
done
# also wait out the eval phase if train just finished
while pgrep -f "eval_grpo_ood.py" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%T)] watcher: GRPO finished, starting LoRA-ES" | tee -a "$LOGDIR/driver.log"

tag=loraes_N16
echo "[$(date +%T)] START $tag" | tee -a "$LOGDIR/driver.log"
$P4_PY track_a/src/es_train_lora.py --model Qwen/Qwen2.5-3B-Instruct --dataset gsm8k \
  --population_size 16 --batch 8 --num_steps 200 --sigma 0.005 --alpha 2e-3 \
  --pop_seed 0 --gpu 0 --eval_every 100 --eval_cap 300 \
  --out_prefix "$RESDIR/$tag" > "$LOGDIR/${tag}.log" 2>&1
echo "[$(date +%T)] DONE $tag rc=$?" | tee -a "$LOGDIR/driver.log"
