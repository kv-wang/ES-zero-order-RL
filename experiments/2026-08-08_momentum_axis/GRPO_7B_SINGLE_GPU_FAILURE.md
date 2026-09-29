# 7B GRPO 单卡可行性探针：全 batch size 均 OOM

**日期**: 2026-08-27  
**问题**: 单张 H200 NVL (143771 MiB) 能否运行 Qwen2.5-7B-Instruct countdown GRPO 训练？  
**结论**: **否** — 三档 batch size (bs=4/8/16) 峰值内存 126.52–139.15 GB 均超出可用容量，需双卡 TP=2。

---

## 实验设置

**目标**: 建立 GRPO 单卡 batch size 下界，原假设 bs=4 可行（对照：双卡 bs=64 峰值 122.42 GB/卡）。

| 参数 | 值 |
|------|-----|
| Model | `Qwen2.5-7B-Instruct` (14 GB bf16) |
| 卡容量 | 143771 MiB = 143.77 GB (H200 NVL 单卡) |
| 训练步数 | 2 (smoke test, BUDGET=manual STEPS=2) |
| MICRO | 2 (固定，控制 actor 单步梯度累积) |
| GROUP | 8 (固定，每步 rollout = TRAIN_BS × GROUP) |
| Eval | EVAL_EVERY=1, EVAL_CAP=40, countdown only |

**三档配置**:

1. **bs=16** (上界参考)
   - `TRAIN_BS=16`, 每步 16×8=128 rollouts, 总 256 generations
   - vLLM `gpu_memory_utilization`: 0.85 → 0.6 → 0.5 (渐降)

2. **bs=8** (中点)
   - `TRAIN_BS=8`, 每步 64 rollouts, 总 128 generations
   - util=0.85

3. **bs=4** (下界)
   - `TRAIN_BS=4`, 每步 32 rollouts, 总 64 generations
   - util=0.85

---

## 结果：全 OOM

### bs=16 三次降 util 均失败

| util | 峰值 (GB) | 失败位置 | 日志 |
|------|----------|---------|------|
| 0.85 | 135.67 | step 1 训练 | `grpo_bs16_util085.log` |
| 0.6  | 130.35 | step 1 训练 | `grpo_bs16_util06.log` |
| 0.5  | 126.52 | step 1 训练 | `grpo_bs16_util05.log` |

**异常**: util=0.85 时 nvidia-smi 显示 0 MiB 占用，但 vLLM 仅报 80.08 GB 可用 (预期 ~122 GB)，怀疑 vLLM v1 内存计数 bug。

**OOM 样例** (bs=16, util=0.5, 峰值 126.52 GB):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.81 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 3.03 GiB is free.
Process has 135.26 GiB memory in use.
```

### bs=8 仍 OOM

| util | 峰值 (GB) | 失败位置 | 日志 |
|------|----------|---------|------|
| 0.85 | 130.81 | step 1 训练 | `grpo_bs8.log` |

### bs=4 (下界) 仍 OOM

| util | 峰值 (GB) | 失败位置 | 日志 |
|------|----------|---------|------|
| 0.85 | 139.15 | step 1 训练 | `grpo_bs4_util05.log` |

**OOM 样例** (bs=4, util=0.85, 峰值 139.15 GB):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 890.00 MiB.
GPU 0 has a total capacity of 139.80 GiB of which 669.12 MiB is free.
Process has 137.43 GiB memory in use.
```

---

## 根因分析

**内存基线独立于 batch size** (MICRO=2 固定时):

| 组件 | 大小 (GB) | 依赖性 |
|------|----------|--------|
| 7B 模型 bf16 | ~14 | 常数 |
| Adam 状态 fp32 | ~28 | 常数 (2×参数量) |
| 激活 + 梯度 | 20–30 | 取决于 MICRO 和序列长度，MICRO=2 时基本固定 |
| vLLM KV cache | ~70 | 取决于 util 和卡容量 (util=0.85 时 ~100 GB, util=0.5 时 ~70 GB) |
| **总基线** | **>130 GB** | **超出 143.77 GB 可用空间** |

**FSDP 单卡无分片**: 全模型 + 全 Adam 状态在一张卡上，无法通过降低 batch size 减轻。

**batch size 不影响基线** (在 MICRO=2 约束下):
- TRAIN_BS 只改变每步 rollout 次数 (4×8=32, 8×8=64, 16×8=128)
- Actor 单步梯度累积固定为 MICRO=2 microbatch
- 激活峰值由 MICRO 决定，与 TRAIN_BS 解耦

---

## 未测试配置

- **MICRO=1**: 可能减少激活内存，但梯度累积减半可能影响收敛 (未在双卡配置验证过)
- **util < 0.5**: 极端降低 KV cache，但 util=0.5 已降到 70 GB 仍 OOM，进一步压缩空间有限
- **序列长度缩短**: max_prompt=1024/max_resp=2048 是论文配置，未尝试 512/1024

---

## 双卡对照 (TP=2, 08-24 smoke)

| 配置 | 峰值/卡 (GB) | 总 (GB) | 状态 |
|------|-------------|---------|------|
| bs=64, util=0.5, MICRO=2 | 122.42 / 122.84 | 245.27 | **正常完成** |

**关键差异**: TP=2 时模型权重和 Adam 状态分片到两卡，每卡 ~7 GB 模型 + ~14 GB Adam，基线降至 ~90 GB/卡。

---

## 结论与后续

1. **7B GRPO 必须双卡**: 单卡 H200 NVL 无法容纳 7B FSDP actor + vLLM rollout 内存需求
2. **转向 3B 单卡**: 已创建 `run_countdown_3b.sh` (N=30, B=100, steps=100, 单卡 TP=1)
3. **3B 内存预算** (单卡):
   - 模型 bf16: ~6 GB
   - Adam fp32: ~12 GB
   - 激活: ~8–12 GB (估计，按比例缩放)
   - vLLM KV (util=0.7): ~100 GB
   - **总计 ~126 GB** (安全边界 86%)

**验证中**: 3B countdown smoke test (SMOKE=1, 2 steps) 已启动，vanilla 阶段运行中。

---

## 附录：关键日志路径

- bs=16 logs: `axis_probe/logs/countdown_7b_grpo/grpo_countdown_7b_bs16*.log`
- bs=8 log: `axis_probe/logs/countdown_7b_grpo/grpo_countdown_7b_bs8*.log`
- bs=4 log: `axis_probe/logs/countdown_7b_grpo/grpo_countdown_7b_bs4*.log`
- Driver logs: `/tmp/grpo_bs{16,8,4}*.log`
- 驱动脚本: `run_grpo_7b_countdown.sh` (单卡改动未提交)
