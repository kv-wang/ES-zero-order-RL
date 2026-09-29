# Countdown @ B=1000 — 中期检查（vanilla, step 17/100）

**日期**：2026-08-22（检查时刻 10:06 UTC）
**状态**：RUNNING — vanilla 臂第 17/100 步；baseaxis 臂尚未启动
**驱动脚本**：`es_bench/axis_probe/run_countdown_paperB.sh`（tmux 会话 `espaperB`，08-21 18:49:26 启动）

本文件记录一次**零 GPU 的中期重分析**：所有数字都是从正在运行的臂已经写出的产物里算出来的，
没有重跑、没有额外生成、没有打断训练。

---

## 1. 问的是什么

三个问题：

1. **B=1000 是否修好了 fitness 可分辨性？** 07-30 的 countdown 在 B=100 下测得 split-half
   ρ=0.51，而全部数学臂在 B=8 下 ρ 精确为 0（`PAPER_FIDELITY_AUDIT.md`、08-14 台账行）。
   B=1000 是本仓库跑过的最大批量，ρ 应该更高——但从未实测过。
2. **中间评测看得到训练在动吗？** 08-21 新加的 `--eval_interval 10`（每 10 步跑全部 7 个数据集，
   结果增量落盘到 `_evalcurve.jsonl`）第一次产出数据，step 10 的点已经可读。
3. **实际吞吐与脚本的成本预估差多少？** 脚本的 60.3h/臂 是从 08-16 的 B=100 单卡测量线性外推的，
   这次是 B=1000 + TP=2，外推未必成立。

## 2. 设置

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 任务 | countdown（ID），另 6 套数学集作 OOD |
| ES | variant=vanilla，N=30，B=1000，steps=100，σ=1e-3，α=5e-4，mini_batch=64 |
| seed | **0（单 seed）** |
| max_tokens | 2048（训练与评测同一上限，07-30 协议） |
| eval_cap | 300 |
| 硬件 | 2×GPU，tensor_parallel_size=2，gpu_memory_utilization=0.85 |
| 每步生成量 | 30 × 1000 = 30,000（约 1.6×10⁷ token/步） |

数据采样口径（与论文的偏离，沿用 08-21 脚本头部说明）：trainer 每步从 1900 行训练池
**有放回**抽 B=1000 并**每步重抽**，论文用一个固定的全集批次。故本轮的目标函数仍在步间移动，
B=1000 只把每步噪声压到 08-16 的 ~1/10，没有消除它。

## 3. 结果

### 3.1 split-half Spearman ρ —— 离线从 `correct_bits` 算出

方法：每步的 `correct_bits` 是 N=30 个 Python 整数，第 b 位表示该成员是否解出该步 CRN 批次的第 b 题。
把 B=1000 个题目下标随机对半分，各半统计每个成员的解题数，对两条长度 30 的向量求 Spearman。
一步一个 ρ。**不需要模型权重，不需要重新生成。**

| 拆分随机种子 | mean ρ | min | max |
|---|---|---|---|
| 0 | 0.811 | 0.603 | 0.897 |
| 1 | 0.830 | 0.682 | 0.930 |
| 2 | 0.826 | 0.667 | 0.925 |

逐步（拆分种子 0）：

```
step  0  0.784    step  6  0.833    step 12  0.867
step  1  0.811    step  7  0.873    step 13  0.815
step  2  0.603    step  8  0.844    step 14  0.746
step  3  0.728    step  9  0.751    step 15  0.854
step  4  0.893    step 10  0.842    step 16  0.865
step  5  0.785    step 11  0.814    step 17  0.897
```

对照历史：数学线 B=8 → ρ ≡ 0.000；07-28 overnight B=200 → ρ 0.31–0.46；
07-30 countdown B=100 → ρ=0.51；**本轮 B=1000 → ρ≈0.81–0.83**。

fitness 取值数同步恢复：30 个成员里有 22–30 个不同的 fitness 值（B=8 时代只有 3 个，上限 9）。

→ **B 是可分辨性的主控变量，这条线在本轮得到迄今最强的一次确认**（跨三个数量级、四个 B 值单调）。

### 3.2 step 10 中间评测 vs 未训练 base

| 数据集 | base | step 10 | Δ |
|---|---|---|---|
| **countdown（ID）** | 0.0100 | **0.0433** | **+0.033（4.3×）** |
| math500 | 0.5133 | 0.5033 | −0.010 |
| gsm8k | 0.7500 | 0.7200 | −0.030 |
| svamp | 0.8167 | 0.8100 | −0.007 |
| minerva_math | 0.0846 | 0.0956 | +0.011 |
| olympiadbench | 0.1500 | 0.1500 | 0.000 |
| amc23 | 0.4000 | 0.3000 | −0.100 |

训练侧 fitness（成员在 B=1000 上的平均 reward）同向上升：
0.026（step 0）→ 0.053（step 10）→ 0.080（step 15）→ 0.076（step 17）。

评测开销实测 **76.6 s**（7 套共 1812 题），对 ~51 min 的一步而言约 2.5% 额外开销 ——
低于选型时按 base 阶段 62.1 s 估的 13%（那时假设步长 8 min）。

### 3.3 吞吐：比脚本预估慢 45%

| | 值 |
|---|---|
| 脚本预估（08-16 的 0.0724 s/generation 线性外推） | 60.3 h/臂 = 36 min/步 |
| **实测** | 15.2 h / 18 步 = **~51 min/步** |
| 外推全程 | **~85 h/臂 → 两臂串行 ~170 h ≈ 7.1 天**（8/29 完成） |

整卡峰值显存（`nvidia-smi` 采样 5437 点，两卡求和）：**241.9 GB**，两卡各 123,875 MiB、利用率 82%。
按 `MEMORY_REPORT.md`（08-13）的口径，只有整卡采样是有效测量。

## 4. 注意事项 / 限制

- **单 seed（seed=0）。** 低于本仓库的 3-seed 规则，不足以支撑任何"哪个变体更好"的结论。
- **只有 vanilla，baseaxis 一步没跑。** 本轮的核心对比（vanilla vs baseaxis）**完全未解决**。
- **运行未完成（17/100 步）。** step 10 的评测点是曲线上的一个点，不是终点；
  countdown 0.0433 不能与 08-16 的 vanilla 终值 0.240 相比。
- **countdown 的 extract_rate 混淆仍在。** 08-15/08-16 均测得 countdown 的 `<answer>` 格式遵从率
  只有 0.50–0.75，故 accuracy 同时包含"不会做"与"没按格式答"，训练后的涨幅有多少来自格式学习
  未拆开。这条对本轮的 4.3× 同样成立。
- **amc23 只有 40 题**，−0.100 = 4 题，落在噪声内，不构成"OOD 受损"的证据。
- **ρ 的拆分是随机的**，故报了三个拆分种子；三者一致（0.811/0.830/0.826），但它们共享同一批
  `correct_bits`，不是独立重复。
- **吞吐外推假设后 83 步与前 18 步同速**，未考虑 KV 形状随生成长度变化。
- 与 08-16（B=100）**不可配对**：B 变了 10 倍，只能比趋势。

## 5. 未解决

1. **vanilla vs baseaxis** —— 本轮的主问题，baseaxis 未启动。
2. **STEPS=100 是否值得跑满** —— 实测成本 170h（7.1 天）对 60.3h/臂 的原计划翻了 1.4 倍。
   已向用户提出三个选项（STEPS=30 → ~50h 总；STEPS=50 → ~85h 总；保持 100 → ~170h），
   未答复前按 STEPS=100 继续。若改，两臂必须用同一 STEPS 才可比。
3. **ρ 高但 accuracy 涨得慢** —— ρ=0.81 说明 fitness 排名可信，却仍是 51 min/步换 0.003 accuracy。
   可分辨性不是唯一瓶颈，α/σ 的量程是下一个嫌疑，但本轮不测。
4. **训练期 ρ 未实时可见** —— 本次是事后离线算的。是否把这 ~10 行加进 trainer 尚未决定
   （加了就得重启当前臂）。

## 6. 复现

```bash
cd experiments/2026-08-08_momentum_axis/es_bench/axis_probe
# ρ 曲线（零 GPU，从 correct_bits 离线算）
python - <<'EOF'
import json, numpy as np
from scipy.stats import spearmanr
rows=[json.loads(l) for l in open('results/countdown_paperB1000_N30/vanilla_N30_B1000_s0.jsonl')]
for r in rows:
    bits, B = r['correct_bits'], r['batch_size']
    perm = np.random.default_rng(0).permutation(B)
    mA = sum(1 << int(i) for i in perm[:B//2]); mB = sum(1 << int(i) for i in perm[B//2:])
    a  = [bin(int(b) & mA).count('1') for b in bits]
    bb = [bin(int(b) & mB).count('1') for b in bits]
    print(r['step'], round(spearmanr(a, bb).correlation, 3))
EOF
```

产物：
- `es_bench/axis_probe/results/countdown_paperB1000_N30/vanilla_N30_B1000_s0.jsonl` —— 每步 fitness / correct_bits
- `es_bench/axis_probe/results/countdown_paperB1000_N30/vanilla_N30_B1000_s0_evalcurve.jsonl` —— 中间评测曲线
- `es_bench/axis_probe/results/countdown_paperB1000_N30/base_summary.json` —— 未训练基线
- `es_bench/axis_probe/logs/countdown_paperB1000_N30/gpu_mem.csv` —— 整卡显存采样
