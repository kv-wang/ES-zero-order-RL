# 七月实验的任务与基座模型审计 (2026-08-13)

**结论:七月的工作一路从 MATH 迁移到 countdown,并在 07-26 起把基座从
`Qwen2.5-Math-1.5B-Instruct` 换成通用版 `Qwen2.5-3B-Instruct`。8 月的 try 因为是从
07-23 拷贝的,继承了最早期的组合,等于跳过了七月后半段的演进——退回到了一个
论文从未测试、且已被自己的数据证明无信号的配置。**

## 问题

八月的全部结果都是 null。是方法问题还是配置问题?七月做过哪些任务、用过哪些基座?

## 方法

不读 `config.py`(驱动脚本可覆盖),而是从**实际产出的 `*summary*.json`** 里读
`model` 字段,并按**文件 mtime** 分组——因为每个 try 都是前一个的完整快照拷贝,
直接按目录统计会把继承来的旧结果算进去。任务由 `eval_final` 的键名推断。

## 时间线

| 时段 | try | 基座 | 任务 |
|---|---|---|---|
| 07-18 ~ 07-19 | 07-18 | Math-1.5B-Instruct、1.5B-Instruct、3B-Instruct | MATH、GSM8K |
| 07-22 ~ 07-23 | 07-21 | **Math-1.5B-Instruct** | MATH(25+ 个 arm) |
| 07-23 ~ 07-24 | 07-23 | **Math-1.5B-Instruct** | MATH(base-axis pilot/confirm) |
| 07-24 ~ 07-25 | 07-24 | **1.5B-Instruct** | MATH(Track A LoRA-ES) |
| 07-25 ~ 07-26 | 07-24 | **3B-Instruct** | GSM8K → countdown |
| 07-27 ~ 07-29 | 07-24 | **3B-Instruct** | countdown、MATH |
| **07-30** | 07-24 | **3B-Instruct** | **countdown**(battery) |

## 三个任务、三个基座,全部是 Instruct 版

| 任务 | 用过的基座 |
|---|---|
| MATH(L3-5 训练) | Math-1.5B-Instruct、1.5B-Instruct、3B-Instruct |
| GSM8K | 3B-Instruct、1.5B-Instruct |
| **Countdown** | **3B-Instruct** |

**从未使用过任何 base(非 Instruct)版模型。**

## 两处关键发现

**一、`Math-1.5B-Instruct` 只在 07-18 ~ 07-23 用过,07-24 起就被弃用了。**
之后全部换成通用版 1.5B / 3B-Instruct。而 8 月的 try 从 07-23 拷贝,
**继承的正是这个已被弃用的选择**。没有记录说明为何弃用,也没有记录说明 8 月为何回到它。

**二、七月的终点是 countdown,不是 MATH。**
07-30 的 countdown battery 是唯一测到 ES 真正学得动的运行:base 0.08 → full-ES 0.450,
split-half ρ = 0.51 持续 100 步。而 8 月的 try 退回 MATH,ρ 中位数为 0。

所以"搬到 countdown"不是新提案,是**七月底已经走到的位置**。

## 与论文配置对照

见 [ES 论文 arXiv:2509.24372](https://arxiv.org/abs/2509.24372):

| 论文实验 | 论文用的基座 | 起点准确率 | 本仓库对应 |
|---|---|---|---|
| Countdown | Qwen-2.5-{0.5,1.5,3,7}B-**Instruct** 等 | 0.1%–31.2% | 07-30 用 3B-Instruct,**与论文一致** |
| 数学推理 | **`Qwen2.5-Math-7B`(基座,非 Instruct)** | — | 本仓库用 Math-1.5B-**Instruct**,**不一致** |
| ARC-AGI | Qwen-2.5-14B | 0.2% | 未做 |
| Sudoku | Qwen-2.5-3B | 2.5% | 未做 |

论文所有实验的起点准确率都在 **0.1%–31.2%** 区间;而 8 月 try 的 MATH L3-5 起点是
**0.6452**。这不是论文任何一个实验所处的区间。

## 含义

八月全部 null 结果现在有了统一解释:**在一个论文从未测试过的 regime 里复现论文的方法**
——起点已被 GRPO 优化(见 [BASE_VS_TRAINED_REPORT](2026-08-08_momentum_axis/BASE_VS_TRAINED_REPORT.md))、
准确率 0.65、split-half ρ ≈ 0。这与方法本身的优劣无关。

## 未解决

**为何 07-24 弃用 `Math-1.5B-Instruct`?** 台账与各 try 的报告中都没有记录这次切换的理由。
可能是有意的(发现该基座不合适),也可能是随手改的。**这条信息本应进 ledger 却没有。**

**为何 8 月回到它?** 因为新 try 从 07-23 拷贝。这是快照约定的一个副作用:
拷贝会连同"当时的配置选择"一起继承,而**不会继承之后的修正**。
建议在 `experiments/README.md` 的"启动新 try"段落里加一句:
从最近的 try 拷贝,而不是从任意一个。

**任务推断有粗糙之处。** 部分 summary 的 `eval_final` 结构不同,归为 `?`;
`mtime` 也可能被拷贝操作改写(虽然实测各 try 的 07-18 批次 mtime 一致,说明拷贝保留了时间戳)。
本审计用于定性判断迁移方向,不宜用作精确计数。
