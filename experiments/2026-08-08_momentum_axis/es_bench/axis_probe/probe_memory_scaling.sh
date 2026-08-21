#!/usr/bin/env bash
set -euo pipefail

# 显存探测：ES (N=30 B=1000) vs GRPO (group ∈ {8,16,32,64})
# 目标：验证 group size 是否影响 GRPO 显存，以及 B=1000 下 ES 的真实峰值
#
# 输出：
#   - 每个配置的 nvidia-smi 峰值 (整卡观测)
#   - torch 峰值 (仅 ES，GRPO 因进程隔离不可比)
#   - 每步耗时与 KV 利用率
#
# 用法：
#   ./probe_memory_scaling.sh               # 跑全部 5 个配置
#   SKIP_ES=1 ./probe_memory_scaling.sh     # 只跑 GRPO 扫描
#   SKIP_GRPO=1 ./probe_memory_scaling.sh   # 只跑 ES

REPO_ROOT=/home/hyin66/es-fine-tuning-paper
TRY_ROOT="$REPO_ROOT/experiments/2026-08-08_momentum_axis/es_bench"
cd "$TRY_ROOT"

# verl 环境解释器（系统 python3 没有 numpy/torch/vllm）
source axis_probe/env.sh

# export，不是普通赋值：config.py 从环境读 MODEL，默认值是 Qwen/Qwen2.5-Math-1.5B。
# 不 export 的话 ES 段会训练 Math-1.5B，而报告标题写着 1.5B-Instruct。
export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TASK=countdown
PROBE_DIR="axis_probe/results/memory_probe_$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="axis_probe/logs/memory_probe"
CKPT_BASE=/tmp/memory_probe_ckpts

mkdir -p "$PROBE_DIR" "$LOGS_DIR" "$CKPT_BASE"

# nvidia-smi 后台采样函数
sample_memory() {
    local pid=$1
    local out_csv=$2
    local interval=2  # 秒

    echo "timestamp_epoch,memory_used_mib" > "$out_csv"

    while kill -0 "$pid" 2>/dev/null; do
        local ts=$(date +%s)
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        echo "$ts,$mem" >> "$out_csv"
        sleep "$interval"
    done
}

run_stage() {
    local stage_name=$1
    local cmd=$2
    local log_path="$LOGS_DIR/${stage_name}.log"
    local mem_csv="$PROBE_DIR/${stage_name}_memory.csv"
    local rc_file="$PROBE_DIR/${stage_name}.rc"

    echo "[$(date +%H:%M:%S)] Starting: $stage_name"
    echo "  Log: $log_path"
    echo "  Memory: $mem_csv"

    # 清空之前的残留
    rm -f "$mem_csv" "$rc_file"

    # 启动训练
    bash -c "$cmd" &> "$log_path" &
    local train_pid=$!

    # 启动显存采样
    sample_memory "$train_pid" "$mem_csv" &
    local sampler_pid=$!

    # 等待训练完成。注意：先捕获 rc 再做任何其他事，
    # `wait ... || true` 会让后面的 $? 恒为 0（08-15 同类缺陷）
    local rc=0
    wait "$train_pid" || rc=$?
    echo "$rc" > "$rc_file"

    # 停止采样
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    # 读取峰值
    if [[ -f "$mem_csv" ]]; then
        local peak_mib=$(tail -n+2 "$mem_csv" | cut -d, -f2 | sort -rn | head -1)
        local peak_gb=$(awk "BEGIN{printf \"%.2f\", ${peak_mib:-0}/1024}")
        echo "  Peak memory: $peak_gb GB (nvidia-smi)"
    else
        echo "  Memory CSV not found"
        peak_gb="N/A"
    fi

    echo "  Exit code: $rc"
    echo ""

    return $rc
}

# ============================================================================
# Stage 1: ES (N=30 B=1000 5步)
# ============================================================================
if [[ "${SKIP_ES:-0}" != "1" ]]; then
    ES_STEPS=5
    ES_N=30
    ES_B=1000
    ES_UTIL=0.85

    # 参数名必须和 es_train_axis.py 的 argparse 一致。旧版本传的
    # --model/--batch_size/--save_dir/--summary_path/--jsonl_path 五个都不存在，
    # argparse 会在任何 GPU 工作之前 exit 2；模型走 MODEL 环境变量（已 export），
    # 输出走单一 --out_prefix（trainer 自己拼 _summary.json 和 .jsonl）。
    # 这是纯显存探测：不加 --eval_final / --kl，省掉评测那一段时间。
    # --mini_batch 64 与 run_countdown_paperB.sh 一致：它决定 vLLM 的单次
    # generate 块大小，也就是 B=1000 下 KV 峰值的真正上界，改了就不是同一个测量。
    ES_CMD="$P4_PY axis_probe/src/es_train_axis.py \
        --dataset countdown \
        --variant vanilla \
        --population_size $ES_N \
        --pop_seed 99 \
        --num_steps $ES_STEPS \
        --batch $ES_B \
        --mini_batch 64 \
        --max_tokens 2048 \
        --gpu 0 \
        --gpu_mem_util $ES_UTIL \
        --out_prefix '$PROBE_DIR/es_N${ES_N}_B${ES_B}'"

    run_stage "es_N${ES_N}_B${ES_B}" "$ES_CMD" || echo "ES failed but continuing"
fi

# ============================================================================
# Stage 2-5: GRPO (group ∈ {8,16,32,64}, 各5步)
# ============================================================================
if [[ "${SKIP_GRPO:-0}" != "1" ]]; then
    GRPO_STEPS=5
    GRPO_BS=16
    GRPO_UTIL=0.5

    for GROUP in 8 16 32 64; do
        ROLLOUT=$((GRPO_BS * GROUP))

        # verl 需要的数据准备（复用 08-16 的 parquet）
        PARQUET_DIR="axis_probe/grpo/data_countdown/${MODEL//\//_}"
        if [[ ! -f "$PARQUET_DIR/train.parquet" ]]; then
            echo "Generating countdown parquet for GRPO (one-time)"
            $P4_PY axis_probe/grpo/make_countdown_parquet.py \
                --model "$MODEL" \
                --out_dir "$PARQUET_DIR"
        fi

        GRPO_CMD="$P4_PY -m verl.trainer.main_ppo \
            data.train_files='$PARQUET_DIR/train.parquet' \
            data.val_files=null \
            data.train_batch_size=$GRPO_BS \
            data.max_prompt_length=1024 \
            data.max_response_length=512 \
            trainer.total_epochs=1 \
            trainer.project_name=memory_probe \
            trainer.experiment_name=grpo_group${GROUP} \
            trainer.n_gpus_per_node=1 \
            trainer.save_freq=-1 \
            trainer.nnodes=1 \
            trainer.logger=['console'] \
            trainer.default_local_dir='$CKPT_BASE/grpo_group${GROUP}' \
            actor_rollout_ref.model.path='$MODEL' \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=$GRPO_BS \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.n=$GROUP \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.gpu_memory_utilization=$GRPO_UTIL \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
            custom_reward_function.path='axis_probe/grpo/reward_countdown.py' \
            custom_reward_function.name=compute_score \
            algorithm.kl_ctrl.kl_coef=0.001"

        # verl 会跑 total_epochs × (数据集大小 / batch_size) 步
        # countdown train 有 1900 行，batch=16 → 119 步/epoch
        # 我们只要 5 步，所以用一个小 subset
        # 但 verl 不支持直接限制步数，只能通过 total_epochs 控制
        # 临时方案：复制前 80 行数据 (80/16=5 步)

        SUBSET_DIR="$PARQUET_DIR/subset_5steps"
        mkdir -p "$SUBSET_DIR"
        $P4_PY << EOPYTHON
import pandas as pd
df = pd.read_parquet("$PARQUET_DIR/train.parquet")
df_subset = df.head(80)  # 80 / 16 = 5 steps
df_subset.to_parquet("$SUBSET_DIR/train.parquet", index=False)
EOPYTHON

        # 更新命令使用 subset
        GRPO_CMD="${GRPO_CMD/train.parquet/subset_5steps\/train.parquet}"

        run_stage "grpo_group${GROUP}_bs${GRPO_BS}" "$GRPO_CMD" || echo "GRPO group=$GROUP failed but continuing"
    done
fi

# ============================================================================
# 汇总报告
# ============================================================================
$P4_PY << 'EOREPORT'
import json
import glob
import csv
from pathlib import Path

probe_dir = Path("axis_probe/results").glob("memory_probe_*")
probe_dir = sorted(probe_dir)[-1]  # 最新的

print("=" * 80)
print(f"显存探测报告 — {probe_dir.name}")
print("=" * 80)
print()

configs = []

# 收集所有配置的数据
for mem_csv in sorted(probe_dir.glob("*_memory.csv")):
    stage_name = mem_csv.stem.replace("_memory", "")
    rc_file = probe_dir / f"{stage_name}.rc"
    summary_file = probe_dir / f"{stage_name}_summary.json"

    # 读取峰值
    with open(mem_csv) as f:
        reader = csv.DictReader(f)
        mem_values = [int(row["memory_used_mib"]) for row in reader]

    peak_mib = max(mem_values) if mem_values else 0
    peak_gb = peak_mib / 1024

    # 读取退出码
    rc = int(open(rc_file).read().strip()) if rc_file.exists() else -1

    # 读取 torch 峰值（仅 ES 有）
    torch_peak_gb = None
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
            torch_peak_gb = summary.get("peak_mem_alloc_gb", 0)

    # 解析配置
    if stage_name.startswith("es_"):
        parts = stage_name.replace("es_N", "").replace("_B", " ").split()
        method = "ES"
        N = int(parts[0])
        B = int(parts[1])
        rollout_per_step = N * B
        group = N
        config_label = f"N={N} B={B}"
    else:  # grpo_
        parts = stage_name.replace("grpo_group", "").replace("_bs", " ").split()
        method = "GRPO"
        group = int(parts[0])
        bs = int(parts[1])
        rollout_per_step = group * bs
        N, B = None, None
        config_label = f"group={group} bs={bs}"

    configs.append({
        "method": method,
        "config": config_label,
        "group": group,
        "rollout_per_step": rollout_per_step,
        "peak_gb": peak_gb,
        "torch_gb": torch_peak_gb,
        "rc": rc,
        "stage": stage_name
    })

# 打印表格
print("Method | Config              | Rollout/step | Peak (smi) | Torch Peak | Status")
print("-" * 85)

for c in configs:
    torch_str = f"{c['torch_gb']:.1f} GB" if c['torch_gb'] else "N/A"
    status = "✓ OK" if c['rc'] == 0 else f"✗ rc={c['rc']}"
    print(f"{c['method']:6s} | {c['config']:19s} | {c['rollout_per_step']:12d} | "
          f"{c['peak_gb']:10.1f} | {torch_str:10s} | {status}")

print()
print("=" * 80)
print("关键发现")
print("=" * 80)

# ES 基线
es_configs = [c for c in configs if c['method'] == 'ES']
if es_configs:
    es = es_configs[0]
    print(f"1. ES (N=30 B=1000) 峰值: {es['peak_gb']:.1f} GB (nvidia-smi)")
    print(f"   单步 rollout: {es['rollout_per_step']:,}")
    print()

# GRPO 扫描
grpo_configs = [c for c in configs if c['method'] == 'GRPO']
if grpo_configs:
    print(f"2. GRPO group 扫描（train_bs=16，util=0.5）:")
    for g in grpo_configs:
        print(f"   group={g['group']:3d}  →  峰值 {g['peak_gb']:6.1f} GB  "
              f"rollout/step={g['rollout_per_step']:5d}")

    # 显存是否随 group 变化
    grpo_peaks = [c['peak_gb'] for c in grpo_configs if c['rc'] == 0]
    if len(grpo_peaks) >= 2:
        peak_range = max(grpo_peaks) - min(grpo_peaks)
        print(f"\n   峰值极差: {peak_range:.1f} GB")
        if peak_range < 5:
            print("   → Group size 对显存影响极小（< 5 GB），假设成立 ✓")
        else:
            print(f"   → Group size 显著影响显存（{peak_range:.1f} GB），假设不成立 ✗")

print()
print("详细数据：")
print(f"  CSV:  {probe_dir}/*_memory.csv")
print(f"  JSON: {probe_dir}/*_summary.json")
print(f"  Logs: axis_probe/logs/memory_probe/")
EOREPORT

echo ""
echo "探测完成。Checkpoint 已保存在 $CKPT_BASE（可删除）"
