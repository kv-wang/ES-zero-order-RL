#!/usr/bin/env bash
# A0 GRPO microbench: full-param (LORA_RANK=0) or LoRA (LORA_RANK=16) GRPO on Qwen-3B, GSM8K,
# 64 rollouts/step (TRAIN_BS=8 x GROUP=8), max_response=256 -> matches LoRA-ES N=8. Short run;
# per-step verl timing_s/* parsed offline. Peak VRAM sampled via nvidia-smi.
set -uo pipefail
cd "$(dirname "$0")/../.."                     # es_bench dir
BENCH=track_a/grpo_bench
PY=/home/hyin66/micromamba/envs/verl/bin/python
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
GPU=${GPU:-0}
LORA_RANK=${LORA_RANK:-0}
STEPS=${STEPS:-200}
TRAIN_BS=${TRAIN_BS:-16}; GROUP=${GROUP:-8}; MICRO=${MICRO:-8}
MAXRESP=${MAXRESP:-512}; MAXPROMPT=${MAXPROMPT:-1024}
EXP=${EXP:-grpo_full}
OUT=/tmp/grpo_ckpt_$EXP; LOG=$BENCH/logs/$EXP.log; VRAM=$BENCH/logs/${EXP}_vram.txt
mkdir -p "$OUT" "$BENCH/logs"

# vLLM rollout utilization MUST match the ES trainers -- see TRAIN_GPU_MEM_UTIL in
# track_{a,b}/src/config.py for why (KV cache -> concurrency -> s/step; unmatched settings make
# cross-method wall-clock uninterpretable). Shell cannot import config.py, so the value is read
# back from it here and the run aborts on drift rather than silently diverging.
CFG=$(dirname "$0")/../src/config.py
GPU_MEM_UTIL=$(grep -oP '^TRAIN_GPU_MEM_UTIL\s*=\s*\K[0-9.]+' "$CFG") \
  || { echo "FATAL: cannot read TRAIN_GPU_MEM_UTIL from $CFG"; exit 1; }
[ -n "$GPU_MEM_UTIL" ] || { echo "FATAL: TRAIN_GPU_MEM_UTIL empty in $CFG"; exit 1; }
echo "rollout gpu_memory_utilization=$GPU_MEM_UTIL (from $CFG, matched to ES)"

export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_DISABLE_XET=1 HF_DATASETS_CACHE=/home/hyin66/.cache/hf_datasets
export ES_GRPO_REWARD_LOG=$BENCH/logs/${EXP}_reward.jsonl; : > "$ES_GRPO_REWARD_LOG"
export VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export XDG_CONFIG_HOME=/home/hyin66/.cache/p4_xdg_config MPLCONFIGDIR=/home/hyin66/.cache/p4_mpl
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

# peak-VRAM sampler
( m=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null|head -1); [ "$u" -gt "$m" ] 2>/dev/null && m=$u && echo $m > "$VRAM"; sleep 1; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

LORA_ARGS=""
if [ "$LORA_RANK" -gt 0 ]; then
  LORA_ARGS="actor_rollout_ref.model.lora_rank=$LORA_RANK actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
    actor_rollout_ref.rollout.load_format=safetensors"
fi

$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=${DATA_DIR:-$BENCH/data}/train.parquet data.val_files=${DATA_DIR:-$BENCH/data}/val.parquet \
  data.train_batch_size=$TRAIN_BS data.max_prompt_length=$MAXPROMPT data.max_response_length=$MAXRESP \
  data.reward_fn_key=data_source data.filter_overlong_prompts=True data.truncation=right \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=$GROUP \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
  custom_reward_function.path=${REWARD_FN:-$(pwd)/grpo/reward_logged.py} custom_reward_function.name=compute_score \
  $LORA_ARGS \
  trainer.logger=[console] trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=${STEPS} trainer.test_freq=-1 trainer.val_before_train=False \
  trainer.total_epochs=1000 trainer.total_training_steps=$STEPS \
  trainer.project_name=a0_grpo_bench trainer.experiment_name=$EXP \
  trainer.default_local_dir=$OUT 2>&1 | tee "$LOG"
echo "peak VRAM (MiB): $(cat $VRAM 2>/dev/null)"
