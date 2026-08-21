#!/usr/bin/env bash
# GRPO control via verl (single GPU). Same model + same shared reward as ES.
# rollout.n = GRPO group size G; train_batch_size prompts/step => G*batch rollouts/step.
# Reward calls are logged to $REWARD_LOG for offline zero-advantage-group analysis.
set -uo pipefail
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock
PY=/home/hyin66/micromamba/envs/verl/bin/python
REPO=/home/hyin66/es-fine-tuning-paper/experiments/2026-07-18_es-smallpop-vs-grpo-wallclock

MODEL=${MODEL:-Qwen/Qwen2.5-Math-1.5B-Instruct}
GPU=${GPU:-1}
STEPS=${STEPS:-200}
TRAIN_BS=${TRAIN_BS:-8}       # prompts/step
GROUP=${GROUP:-8}             # rollout.n (GRPO group size) -> TRAIN_BS*GROUP rollouts/step
MINI_BS=${MINI_BS:-8}         # ppo_mini_batch_size (prompts)
MICRO=${MICRO:-8}             # per-gpu micro batch (responses)
LR=${LR:-1e-6}
MAXPROMPT=${MAXPROMPT:-1024}
MAXRESP=${MAXRESP:-512}
EXP=${EXP:-phase1_grpo}
OUTDIR=${OUTDIR:-$REPO/es_bench/grpo/out_$EXP}
REWARD_LOG=${REWARD_LOG:-$REPO/es_bench/grpo/reward_${EXP}.jsonl}
mkdir -p "$OUTDIR"
: > "$REWARD_LOG"   # truncate

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE=/home/hyin66/.cache/hf_datasets
export ES_GRPO_REWARD_LOG=$REWARD_LOG
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
# ~/.config is a root-owned FILE on this node; redirect XDG + disable vllm usage stats
export XDG_CONFIG_HOME=/tmp/xdgconfig
export MPLCONFIGDIR=/tmp/mplconfig
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"

$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$REPO/es_bench/grpo/data/train.parquet \
  data.val_files=$REPO/es_bench/grpo/data/val.parquet \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=$MAXPROMPT \
  data.max_response_length=$MAXRESP \
  data.reward_fn_key=data_source \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=$GROUP \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO \
  custom_reward_function.path=$REPO/es_bench/grpo/reward_logged.py \
  custom_reward_function.name=compute_score \
  trainer.logger=[console] \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1000 \
  trainer.total_training_steps=$STEPS \
  trainer.project_name=es_grpo \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir=$OUTDIR
