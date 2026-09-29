#!/usr/bin/env bash
# GRPO 显存扫描：Qwen2.5-7B-Instruct，countdown，GROUP=8 固定，TRAIN_BS 扫描，双卡 FSDP。
#
# 问题：TRAIN_BS 增大时 GRPO 的整卡显存峰值怎么变？
#
# 为什么用 nvidia-smi 而不是 torch 计数器：GRPO 的 vLLM 跑在独立进程，verl 的 actor 指标在
# TaskRunner 侧读，`torch.cuda.max_memory_allocated()` 按进程统计因此看不见 KV cache
# （2026-08-13 MEMORY_REPORT 的核心教训）。整卡采样绕开进程布局问题。
#
# 与 08-16 countdown GRPO 的一致处：数据/奖励/恒等模板/GROUP=8/lr/kl_coef 全同；
# 不同处：MODEL 7B（那轮 1.5B-Instruct）、双卡（那轮单卡）、只跑 STEPS 步（那轮 2343 步）。
# 因此本脚本测的是**显存**，不产生任何准确率结论。
#
# 用法：
#   ./probe_grpo_7b_bs.sh                     # BS ∈ {4,8,16,32}
#   BS_LIST="8 16" ./probe_grpo_7b_bs.sh
#   STEPS=2 ./probe_grpo_7b_bs.sh
set -uo pipefail
cd "$(dirname "$0")/.."                       # es_bench
source axis_probe/env.sh

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BS_LIST="${BS_LIST:-4 8 16 32}"
GROUP="${GROUP:-8}"
STEPS="${STEPS:-3}"
MAXPROMPT="${MAXPROMPT:-1024}"
MAXRESP="${MAXRESP:-2048}"        # = 7B ES 臂的 max_tokens，KV 峰值由它定上界
MICRO="${MICRO:-2}"               # per-GPU micro batch；7B 上 8 会 OOM，先保守
ROLLOUT_MEM="${ROLLOUT_MEM:-0.5}"
NGPU="${NGPU:-2}"
GPUS="${GPUS:-0,1}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
INTERVAL="${INTERVAL:-1}"

TS=$(date +%Y%m%d_%H%M%S)
OUT="axis_probe/results/grpo_mem_7b_$TS"
LOG="axis_probe/logs/grpo_mem_7b_$TS"
CKPT=/tmp/grpo_mem_7b_$TS
mkdir -p "$OUT" "$LOG" "$CKPT"
SUMMARY="$OUT/summary.jsonl"

DATADIR=axis_probe/grpo/data_countdown
RAWTMPL="/tmp/rawtmpl_$(basename "$MODEL")"
[[ -f "$RAWTMPL/tokenizer_config.orig.json" ]] \
  || $P4_PY axis_probe/grpo/make_raw_template_model.py "$MODEL" "$RAWTMPL" 2>&1 | tee "$LOG/rawtmpl.log"

# 每卡至少要空出来，否则峰值不可归因（gpu_peak.py 同一判据）
for g in ${GPUS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [[ "$used" -gt 2000 ]]; then
    echo "!! GPU $g 已占用 ${used} MiB，峰值不可归因于本次测量。中止。"; exit 2
  fi
done

echo "model=$MODEL group=$GROUP steps=$STEPS bs_list='$BS_LIST' ngpu=$NGPU tp=$ROLLOUT_TP"
echo "maxprompt=$MAXPROMPT maxresp=$MAXRESP micro=$MICRO rollout_mem=$ROLLOUT_MEM"
echo "out=$OUT"

for BS in $BS_LIST; do
  TAG="bs${BS}_g${GROUP}"
  MEM="$OUT/${TAG}_mem.csv"
  # verl 会跑 total_epochs × ceil(rows/BS) 步，没有直接的步数上限开关，
  # 所以用 total_training_steps 截断（run_grpo.sh 同一做法）。
  echo "timestamp_epoch,gpu,mem_used_mib" > "$MEM"

  ( while :; do
      ts=$(date +%s)
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPUS" \
        | while IFS=', ' read -r idx m; do echo "$ts,$idx,$m" >> "$MEM"; done
      sleep "$INTERVAL"
    done ) &
  SAMPLER=$!

  T0=$SECONDS
  CUDA_VISIBLE_DEVICES=$GPUS $P4_PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATADIR/train.parquet" \
    data.val_files="$DATADIR/val.parquet" \
    data.train_batch_size=$BS \
    data.max_prompt_length=$MAXPROMPT data.max_response_length=$MAXRESP \
    data.reward_fn_key=data_source data.filter_overlong_prompts=True data.truncation=right \
    actor_rollout_ref.model.path="$RAWTMPL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BS \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$GROUP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_MEM \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
    custom_reward_function.path=axis_probe/grpo/reward_countdown.py \
    custom_reward_function.name=compute_score \
    trainer.logger=[console] trainer.n_gpus_per_node=$NGPU trainer.nnodes=1 \
    trainer.default_local_dir="$CKPT/$TAG" \
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=False \
    trainer.total_epochs=1000 trainer.total_training_steps=$STEPS \
    > "$LOG/${TAG}.log" 2>&1
  RC=$?          # 先捕获，$(date) 会把 $? 重置为 0（08-15 缺陷）
  EL=$((SECONDS - T0))
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null

  $P4_PY - "$MEM" "$SUMMARY" <<PY
import csv, json, sys, collections
mem, out = sys.argv[1], sys.argv[2]
per = collections.defaultdict(list)
for r in csv.DictReader(open(mem)):
    try: per[r["gpu"]].append(int(r["mem_used_mib"]))
    except (ValueError, KeyError): pass
peaks = {g: max(v) for g, v in per.items() if v}
tot = collections.defaultdict(int)
for r in csv.DictReader(open(mem)):
    try: tot[r["timestamp_epoch"]] += int(r["mem_used_mib"])
    except (ValueError, KeyError): pass
rec = {
    "tag": "$TAG", "train_batch_size": $BS, "group": $GROUP,
    "rollouts_per_step": $BS * $GROUP, "steps": $STEPS, "rc": $RC, "elapsed_s": $EL,
    "n_gpus": $NGPU, "rollout_tp": $ROLLOUT_TP, "micro": $MICRO,
    "max_prompt": $MAXPROMPT, "max_resp": $MAXRESP, "rollout_mem_util": $ROLLOUT_MEM,
    "peak_per_gpu_mib": peaks,
    "peak_per_gpu_gib": {g: round(v / 1024, 2) for g, v in peaks.items()},
    "peak_max_single_gpu_mib": max(peaks.values()) if peaks else 0,
    "peak_both_cards_mib": max(tot.values()) if tot else 0,
    "peak_both_cards_gib": round(max(tot.values()) / 1024, 2) if tot else 0,
    "samples": sum(len(v) for v in per.values()),
}
open(out, "a").write(json.dumps(rec) + "\n")
print(f"  bs={rec['train_batch_size']} rollouts/step={rec['rollouts_per_step']} rc={rec['rc']} "
      f"{rec['elapsed_s']}s  peak/card={rec['peak_max_single_gpu_mib']/1024:.2f} GiB  "
      f"both={rec['peak_both_cards_gib']} GiB")
PY

  if [[ $RC -ne 0 ]]; then
    echo "  !! bs=$BS rc=$RC（可能是 OOM）；已记录，继续下一档。见 $LOG/${TAG}.log"
    grep -m2 -i "out of memory\|CUDA error" "$LOG/${TAG}.log" | sed 's/^/     /'
  fi
  rm -rf "$CKPT/$TAG"
  sleep 20                                  # 等显存归还，否则下一档的 baseline 不干净
done

echo
echo "== 汇总 =="
$P4_PY -c "
import json,sys
for l in open('$SUMMARY'):
    d=json.loads(l)
    print(f\"bs={d['train_batch_size']:<3} rollouts={d['rollouts_per_step']:<4} rc={d['rc']} \"
          f\"peak/card={d['peak_max_single_gpu_mib']/1024:6.2f} GiB  both={d['peak_both_cards_gib']:7.2f} GiB  \"
          f\"{d['elapsed_s']}s\")
"
echo "summary: $SUMMARY"
