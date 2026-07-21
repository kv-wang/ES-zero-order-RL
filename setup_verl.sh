#!/usr/bin/env bash
# Install verl + vllm into the dedicated, PERSISTENT env at
# /home/hyin66/micromamba/envs/verl. Build/pip caches go to ephemeral /tmp
# so only the final env consumes the 21GB persistent JuiceFS quota.
set -uo pipefail

PY=/home/hyin66/micromamba/envs/verl/bin/python
PIP="$PY -m pip"

# --- keep transient junk OFF the persistent mount ---
export PIP_CACHE_DIR=/tmp/verl_pipcache
export TMPDIR=/tmp/verl_build
export HF_HUB_DISABLE_XET=1                      # avoid the root-owned ~/.cache/hf xet dir
export HF_XET_CACHE=/home/hyin66/.cache/hf_xet
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
disk(){ df -h /home/hyin66 | tail -1 | awk '{print "  persistent /home free: "$4" ("$5" used)"}'; }

log "START verl install"; disk

log "pip/setuptools/wheel upgrade"
$PIP install -q -U pip setuptools wheel 2>&1 | tail -2

# vllm 0.11.0 pulls a matching torch 2.8.0+cu128 and a compatible transformers (4.x)
log "installing vllm==0.11.0 (brings torch 2.8+cu128) ..."
$PIP install "vllm==0.11.0" 2>&1 | tail -4
log "vllm step done"; disk

# verl 0.8.0 with the vllm + math extras; pin numpy<2 as verl requires
log "installing verl[vllm,math]==0.8.0 + numpy<2 ..."
$PIP install "verl[vllm,math]==0.8.0" "numpy<2.0.0" math-verify 2>&1 | tail -6
log "verl step done"; disk

# flash-attn: no nvcc on this machine, so use Dao-AILab's prebuilt wheel for
# torch 2.8 + cu12 + py3.10 + cxx11abiTRUE (matches torch.cuda ABI here).
FLASH_ATTN_WHEEL=${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}
log "installing flash-attn from prebuilt wheel ..."
$PIP install "$FLASH_ATTN_WHEEL" 2>&1 | tail -6
log "flash-attn step done"; disk

log "=== import smoke test ==="
$PY - <<'EOF'
import importlib
for m in ["torch","vllm","verl","transformers","ray","tensordict","numpy","datasets","math_verify","flash_attn"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"ERR {m:14s} {type(e).__name__}: {e}")
import torch
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
print("cxx11_abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
EOF

log "DONE"; disk
