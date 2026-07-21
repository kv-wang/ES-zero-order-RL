#!/usr/bin/env bash
# GRPO | Qwen2.5-3B-Instruct | FSDP actor + vLLM rollout | 2x H200
#
# Prerequisites:
#   - micromamba env `verl` (see ../setup_verl.sh) with flash-attn installed
#   - parquet data at TRAIN_FILE / VAL_FILE (item 2: not prepared yet)
#
# Checkpoints land on ephemeral local disk under /tmp (not JuiceFS).
#
# Example:
#   bash countdown/run_grpo_qwen2_5_3b.sh
#   TRAIN_BATCH_SIZE=32 ROLLOUT_N=4 bash countdown/run_grpo_qwen2_5_3b.sh

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/hyin66/micromamba/envs/verl/bin/python}

# Keep HF / Ray / torch caches off the persistent JuiceFS quota when possible.
export TMPDIR=${TMPDIR:-/tmp/verl_run}
export HF_HOME=${HF_HOME:-/tmp/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/tmp/hf_hub}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$RAY_TMPDIR"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}

# Expected layout after data prep (item 2). Override if you place files elsewhere.
TRAIN_FILE=${TRAIN_FILE:-/tmp/countdown_data/train.parquet}
VAL_FILE=${VAL_FILE:-/tmp/countdown_data/test.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

# Match ES countdown generation budget (max_new_tokens=1024).
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# TP=1 on 2 GPUs (hybrid engine colocates actor + rollout).
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-5}

PROJECT_NAME=${PROJECT_NAME:-countdown_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_3b_instruct}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

# Ephemeral checkpoints (JuiceFS is quota-constrained).
CKPT_DIR=${CKPT_DIR:-/tmp/verl_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# console-only by default; set LOGGER='["console","wandb"]' if wandb is configured.
LOGGER=${LOGGER:-'["console"]'}
########################### end user-adjustable ###########################

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "ERROR: train parquet not found: ${TRAIN_FILE}" >&2
  echo "       Convert countdown/data/countdown.json first (data prep item 2)." >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "ERROR: val parquet not found: ${VAL_FILE}" >&2
  echo "       Convert countdown/data/countdown.json first (data prep item 2)." >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"

# Prefer the verl env interpreter when invoked via bare `python3` from PATH.
export PATH="$(dirname "${PY}"):${PATH}"

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.norm_adv_by_std_in_grpo=True
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=error
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.rollout.enable_chunked_prefill=False
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
)

TRAINER=(
  trainer.critic_warmup=0
  trainer.logger="${LOGGER}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=False
)

# Optional: after wiring countdown_task.reward_function (item 2), pass e.g.
#   reward.custom_reward_function.path=${REPO_ROOT}/countdown/verl_reward.py
#   reward.custom_reward_function.name=compute_score

cd "${REPO_ROOT}"
"${PY}" -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}" \
  "$@"
