#!/usr/bin/env bash
# Phase 4 locked runtime environment. `source` this before any Phase 4 run.
# Certified ES env is VERL (phase1 logs show envs/verl paths): datasets 5.0.0,
# transformers 4.57.6, vllm 0.11 — required for byte-identical Q1 reproduction.
export P4_PY=/home/hyin66/micromamba/envs/verl/bin/python
export HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE=/home/hyin66/.cache/hf_datasets
export XDG_CONFIG_HOME=/home/hyin66/.cache/p4_xdg_config   # writable (~/.config had a non-dir entry)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$XDG_CONFIG_HOME"
