#!/usr/bin/env bash
export P4_PY=/home/hyin66/micromamba/envs/verl/bin/python
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE=/home/hyin66/.cache/hf_datasets
export XDG_CONFIG_HOME=/home/hyin66/.cache/p4_xdg_config
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$XDG_CONFIG_HOME"
