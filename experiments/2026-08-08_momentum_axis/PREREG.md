# 动量轴探针 (A1) — 预注册 (冻结 2026-08-08)

在现有 base-axis 探针旁**并列**增加一个变体:同样的对偶对中心差分估计器,
指向**近期轨迹**而非被取反的全历史。三个 arm 手动选择,除轴之外完全一致。

## 动机

由恒等式 `θ_t = θ₀ + Σ_s Δ_s`:

```
base 轴 = θ₀ − θ_t = − Σ_{s<t} Δ_s
```

即现有探针每步花 4 个成员在问「要不要**撤销全部动量**」。撤销按定义不是学习,
这与 `BESTSHOT_COMPARISON_REPORT` 的结论(锚定"能封住损失,造不出增益")一致。

A1 保留估计器、只换窗口:`m_t = β·m_{t−1} + Δ_t`,有效窗口 ≈ 1/(1−β)。
因为锚定成员是对偶的(±a),符号是装饰性的,所以 **A1 与 base-axis 唯一实质区别是窗口长度 k**。

统计依据:单步更新 `Δ_s = signal + noise`。累加 k 步后信号 ~k、噪声 ~√k,
信噪比提升 ~√k。动量轴因此是一个**方差缩减后的"ES 一直在往哪走"的估计**,
直接针对 split-half ρ = 0.06 这个瓶颈。上限在于损失曲面弯曲会让理想方向旋转,
所以 β 存在最优值——这与经典动量里 μ 的权衡相同。

## 机制

| | 轴 v | scale | 半径 a 的含义 |
|---|---|---|---|
| momentum (A1) | `m_t`(EMA) | `σ√d / ‖m‖` | 一个高斯成员步长的几分之几 |
| baseaxis | `θ₀ − θ_t` | 1 | 向 θ₀ 插值的比例,a=1 正好落在 θ₀ |
| vanilla | — | — | — |

统一系数推导(见 `es_train_axis.py` 头部):成员扰动为 `c·v` 时,

```
Δ_probe = (α/(Nσ)) · Σ_k a_k · scale · (z_{k,+} − z_{k,−}) · v
成员系数 = sign · a_k · scale
```

动量轴必须重归一化到 `σ√d`,因为 `m_t` 没有 a=1 这样的地标。
不做这一步会重演 2026-07-23 的 smoke 故障:探针比 σ√d 小 27–123 倍,`f₊−f₋` 恒为 0。

## 刻意**未**修改的部分(为使对比只隔离轴)

- `a1, a2` 独立同分布抽取(可能相撞;两个半径都记入日志,可事后测量)
- 门控以下**跳过**探针成员而非置零(种群构成在门控处跳变;`use_probe` 每步记录)
- 二元奖励,无连续 tiebreaker(B=8 时约 67% 的对打平)

这三项都是已知缺陷。修它们属于另一个实验,不能和换轴混在一起。

## 已添加的部分(三个 arm 完全对称,不偏袒任何一方)

- **per-member `fitness` / `z` / `gauss_seeds` 写入 JSONL**。07-23 的训练器只记聚合量,
  导致所有事后重分析都无法进行(包括那个能消灭 N=12 混淆的投影对照)。
- **`signed_disp_frac = signed_disp / axis_norm`**。原始 `cum_disp` 跨步不可比,
  因为 `‖axis‖` 随训练增长——4-seed 重分析里 seed 2/3 位移更大,至今分不清是
  锚定更强还是漂移更多。
- `pair_tie_rate`、`zero_update_rate`、`steps_probe_active` 进入 summary。

## 配置

沿用 2026-07-23 冻结配置(`config.py`):Qwen2.5-Math-1.5B-Instruct、MATH L3–5、B=8、
σ=1e-3、α=5e-4、200 步、fp16、CRN。N=16、a_max=1.0。
新增:`β=0.9`(有效窗口 ≈10)、`mom_warmup=5`。

评测:6 套电池(cap 300)+ KL-to-base 代理,与 pilot 完全一致。
因此**本 try 的 baseaxis 与 vanilla arm 可以直接和
`2026-07-23_lora/.../base_axis_probe/results/{pilot,confirm}` 合并**——
CPU 回归测试已证明 baseaxis 路径与冻结 worker 逐位一致。

## 假设与判据

**假设。** 若 ES 的漂移中存在可提取的一致方向,则沿近期动量的方向导数
应当比沿全历史反向的更常非零,且更常为正。

**GO。** 在匹配的 ID 准确率下,momentum 的 `mean_pair_fdiff` 显著大于 baseaxis,
**且** `cum_disp_frac` 为正,**且** OOD 不劣于 vanilla。

**NEGATIVE(照实报告)。** `pair_tie_rate` 与 baseaxis 无差异(≈67%),
或 `cum_disp_frac` ≈ 0 —— 说明近期轨迹同样没有可提取信号。
**不要靠调 β 或 a_max 去抢救**;那属于下一个预注册。

## 已知的、本实验无法回答的问题

1. **regime。** 本实验仍跑在 Qwen2.5-Math-1.5B-Instruct / MATH L3–5,而 countdown battery 表明
   该 regime 的 split-half ρ ≈ 0。若三个 arm 全部 null,首要解释应是 regime 而非机制。
   **countdown 上的重跑仍是更高优先级的实验**,此处沿用 MATH 只是为了与既有 4 个 seed 可比。
2. **N=12 对照**仍然缺失。探针占用 4/16 成员这一混淆对 momentum 同样成立。
3. 单 seed 起步(默认 `SEEDS=0`),未满足 3-seed 规则;任何跨 arm 结论都需先补齐。

## 运行

```bash
cd experiments/2026-08-08_momentum_axis/es_bench
./axis_probe/run_three_arms.sh                  # seed 0,顺序 A1 → base → vanilla,单卡约 7h
SEEDS="0 1 2" ./axis_probe/run_three_arms.sh    # 3 seed,约 21h
N=8 STEPS=20 ./axis_probe/run_three_arms.sh     # smoke
```

已完成 `*_summary.json` 的 arm 会被跳过,被 kill 的运行可直接重跑脚本续上。
