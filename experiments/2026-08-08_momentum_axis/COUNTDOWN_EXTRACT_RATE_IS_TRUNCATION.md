# countdown 的 extract_rate 量的是截断，不是格式遵从

**日期**: 2026-08-31 **状态**: PARTIAL（ES 两臂无 checkpoint，无法探测）
**产物**: `es_bench/axis_probe/probe_countdown_truncation.py`、
`es_bench/axis_probe/results/countdown_3b_truncation_probe/{base,grpo}_trunc.jsonl`

## 问题

08-16 起，本仓库每份 countdown 报告都带同一条免责声明：countdown 准确率是**下界**，
因为 extract_rate 从 base 的 0.890 掉到训练后的 0.51–0.71，"非格式作答得 0 分，所以
准确率混淆了『不会做』与『没按 `<answer>` 格式答』"。3B 线上 GRPO 跑完后这条声明再次出现
（extract 0.890 → 0.7067）。

本次要回答的是：**这个"格式遵从崩塌"的诊断对不对？GRPO 也降抽取率，为什么？**

## 判据本身

`eval_core.py:51` 把抽取定义成一行：

```python
ext = int("<answer>" in txt and "</answer>" in txt)
```

它不检查任何格式结构，只检查这两个标签在不在。一个还在 `<think>` 里搜索、被
`max_tokens=2048` 砍断的回答，和一个完整作答但不用标签的回答，在这个判据下**不可区分**。
`eval_core` 写的逐题 JSONL 既不存原文也不存 token 数，所以从磁盘上的产物无法拆开这两种情形。

## 设置

新脚本 `probe_countdown_truncation.py` 用**完全相同**的评测条件重跑同一批题，额外记录每题的
`n_tokens` 与 vLLM 的 `finish_reason`：

| | |
|---|---|
| 题目 | `countdown/data/countdown.json[:300]`，即钉死的评测切片（训练用 `[300:]`，天然不相交） |
| 提示词 | 数据集自带的原始 `context`（以 `<think>` 开头，不套 chat template） |
| 解码 | greedy（temperature=0），`max_tokens=2048`，`stop=C.STOP` |
| vLLM | `dtype=float16, gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=False` —— 逐项对齐 `eval_grpo.py:73` |
| 评分 | `countdown_task.answer_reward_function`，与 ES 训练器、GRPO 奖励、eval_core 同一函数 |
| 模型 | ① `Qwen/Qwen2.5-3B-Instruct`（base）② `/var/tmp/es_ckpts/grpo_countdown_3b_bs64g8_gen586/global_step_586/hf_merged`（GRPO 末态合并权重） |

**探针复现了两个臂的 summary 数字**：base acc 0.0800 / extract 0.8900，GRPO acc 0.6867 /
extract 0.7067，与 `*_summary.json` 逐位一致。所以下面的 token 统计描述的就是产生那两个数字的
同一批生成，不是另一次运行。

## 结果

### 1. 没有任何一次"格式丢失"

| 臂 | 截断且有标签 | 截断且无标签 | 正常结束且有标签 | **正常结束且无标签** |
|---|---:|---:|---:|---:|
| base | 66 | 33 | 201 | **0** |
| GRPO | 4 | 88 | 208 | **0** |

两个臂都是：**凡是正常终止的回答，100% 闭合了 `<answer>` 标签**（base 201/201，GRPO 208/208）；
**凡是没有标签的回答，100% 是撞上了 2048 token 上限**（base 33/33，GRPO 88/88）。
无标签的回答里，`</think>` 也一个都没闭合，`<answer>` 开了没关的情况为 0。

即：**格式遵从率在两个臂上都是 100%**。extract_rate 不是格式指标，它是
"搜索在预算内收尾了吗"的指标。

### 2. 训练几乎没有改变截断率

| 臂 | 截断率 | n_tokens 中位 | n_tokens 均值 | p75 |
|---|---:|---:|---:|---:|
| base | 0.330 | 421 | 907.9 | 2048 |
| GRPO | 0.307 | 435 | 911.5 | 2048 |

两个臂跑满上限的比例、平均输出长度几乎一样（均值差 0.4%）。**"训练让模型变啰嗦"不成立。**

### 3. 变的是模型在一次失败搜索里做什么

关键列是 `ext | 截断`：

| 臂 | 截断的回答中仍闭合了标签的比例 |
|---|---:|
| base | 66/99 = **0.667** |
| GRPO | 4/92 = **0.043** |

base 的行为是：早早猜一个（通常是错的）答案写进 `<answer>...</answer>`，然后继续絮叨到上限——
所以它三分之二的截断回答里仍然有一个完整标签，抽取率因此看起来很高，但那些标签里的答案基本是错的
（base 抽取内条件正确率仅 0.0899）。

GRPO 的行为是：**在 `<think>` 里一直试，直到验证出一个解才收口**。搜索没成功就不写答案，
于是超预算 = 无标签。它抽取内条件正确率 0.9717，正常结束的 208 题里 206 题是对的。

### 4. 这正是奖励函数要求的行为

ES 与 GRPO 在 countdown 上用的是**同一个纯二值答案奖励**：
`axis_probe/grpo/reward_countdown.py` 直接调 `answer_reward_function`，
`countdown_task.reward_function` 里那个 `0.1 * format_reward` 分项**两个方法都没用**。

在这个奖励下，"写一个错答案"和"不写答案"都得 0 分，完全无差别；而继续搜索至少还有找到解的概率。
所以**没有任何梯度压力促使模型早交卷，早交卷反而放弃了剩余的搜索机会**。延迟承诺是最优策略，
两个方法都学到了它。抽取率下降是奖励设计的直接后果，不是训练损坏了模型。

### 5. 降幅全部来自 4 数题

| 臂 | 3 数题 extract | 4 数题 extract | 3 数题截断率 | 4 数题截断率 |
|---|---:|---:|---:|---:|
| base | 0.9241 | 0.8581 | 0.297 | 0.361 |
| GRPO | **0.9586** | **0.4710** | 0.048 | 0.548 |
| ES vanilla | 0.8138 | 0.3290 | — | — |
| ES baseaxis | 0.8207 | 0.2968 | — | — |

GRPO 在 3 数题上抽取率**上升**了（0.9241 → 0.9586，截断率 0.297 → 0.048）：题目它会做，
搜索很快收敛，收口更干净。整体 extract_rate 下降 100% 来自 4 数题（搜索空间大约两个量级），
那里搜索经常在 2048 token 内跑不完。

### 6. 用不完预算的题，是谁都不会做的题

- GRPO 无标签的 88 题里，base 做对了 **0** 题。
- base 无标签的 33 题里，GRPO 做对了 **19** 题。

所以 extract_rate 的下降没有掩盖任何"本来能做对却被格式吃掉"的题。GRPO 在跑满预算的题上，
base 一题都没解出来。

### 7. ES 臂：机制一致但未实测

ES 两臂**没有 checkpoint**（`es_worker.py:80-83` 有 `es_save_base`，trainer 从未接线），
所以同一探针跑不了。间接证据一致但不构成测量：

- 同一个纯二值奖励，同一套评测代码路径；
- countdown 生成耗时 ES 17.4/17.6 s > GRPO 15.6 > base 14.8；
- ES vanilla 训练曲线 11 个点上，**corr(extract_rate, gen_seconds) = −0.895**，
  corr(extract_rate, accuracy) = −0.867 —— 抽取率随准确率上升而单调下降，
  0.89 → 0.56，抽取内条件正确率同步 0.0899 → 0.8344。

三个训练臂"丢标签"的题高度重合（Jaccard：ES van vs baseaxis 0.565、ES van vs GRPO 0.521），
而与 base 只有 0.10–0.135 —— 三个臂在**同一批难题**上超预算，不是各自随机地损坏格式。

## 结论

1. **`extract_rate` 是命名错误的指标**。它测的是"搜索在 token 预算内收尾的比例"，
   不是格式遵从。两个实测臂的格式遵从都是 100%。
2. **GRPO 确实降抽取率（0.890 → 0.7067），但降幅只有 ES 的一半多**（ES 0.5633），
   原因不是 GRPO"更守格式"，而是它**更会做题**：搜索成功得更早、更频繁，因而更常在预算内收口。
   抽取率在这里实际上是能力的代理量，不是格式的代理量。
3. **仓库沿用了 5 个月的那条免责声明需要改写**。"countdown 准确率是下界，因为格式遵从崩塌"
   —— 诊断错了。它确实是下界，但边界来自 **token 预算**，不是格式。对应的干预手段是
   调 `max_tokens`，不是加格式奖励分项。

## 局限

- **单 seed，单切片**：countdown[:300] 一次，两个臂各一次。
- **ES 两臂未实测**，机制由类比推断（见 §7）。ES 的 extract_rate 0.5633 到底是不是同一机制，
  在 `es_save_base` 接线之前无法回答。
- **只探了 GRPO 末态**：verl 的 in-memory 曲线不记录 extract_rate，GRPO 的抽取率轨迹没有数据，
  无法说它是何时、以什么形状下降的。
- **"加大预算能否把这 88 题救回来"未测**。本报告只证明它们是被砍断的，没有证明放宽 `max_tokens`
  会让它们变成正确答案——也可能只是把悬崖往后挪。
- vLLM 非确定性一般在 ±1–2 点；本次两个臂都逐位复现了 summary，但这不保证下一次也如此。
- 探针复用 `eval_grpo.py:73` 的 vLLM 配置，与 ES 臂 in-process 评测（util=0.70）不同；
  这影响不了贪心解码的输出，但 base 的两次测量能逐位一致本身也说明了这点。

## 未解决

1. `max_tokens` ∈ {2048, 4096, 8192} 的剂量-反应：截断的 88 题里有多少能转化为正确答案？
   这是唯一能把"下界"量化的实验，成本约 3 × 20 分钟（仅 GRPO 末态权重，GPU 已空）。
2. ES trainer 接上 `es_save_base`，否则 ES 侧的任何事后探测都做不了——这是本轮被迫留白的直接原因。
3. countdown 评测是否应当把 `finished_rate` 与 `extract_rate` 分开报告，
   并在 `_eval_countdown` 里记录 `n_tokens` / `finish_reason`（探针已实现，尚未并入 eval_core）。
4. 是否应当启用 `countdown_task.reward_function` 自带的 `0.1 * format` 分项。论文的任务定义里有它，
   本仓库 ES 与 GRPO 都没用。它会给"早交卷"一个正的梯度，可能改变这里描述的全部行为——
   也可能只是教会模型交白卷骗 0.1 分。
