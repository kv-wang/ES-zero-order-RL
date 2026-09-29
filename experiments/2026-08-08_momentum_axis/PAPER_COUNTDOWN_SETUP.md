# 论文 countdown 实验设置 vs 本仓库设置（2026-08-22）

## 问题

正在跑的 `run_countdown_paperB.sh`（Qwen2.5-1.5B-Instruct, N=30, B=1000, STEPS=100, 预计 ~7 天）
声称是"paper-scale"配置。这份文档回答：**论文（arXiv:2509.24372, "Evolution Strategies at
Scale", Qiu et al., ICML 2026）的 countdown 实验到底怎么设的？我们的数据集和它是不是同一个？
还有哪些没对齐的地方？**

## 来源与可信度

- **论文正文/表格**：`https://arxiv.org/html/2509.24372v3`（HTML 版）。附录 A.1/A.3 被抓取工具
  截断，未能读到 → **迭代数、每成员评测题数、max_tokens、reward 公式在正文中没有给出**。
- **官方发布代码**：`countdown/es_fine-tuning_countdown.py`（本仓库根目录自带，来自作者的
  release）。上面那些正文缺失的超参，**全部能从代码里直接读到**，本文以代码为准并标注行号。
- 两者不冲突的部分互相印证（N=30 / σ=0.001 / α=5e-4 / greedy 三处一致）。

## 一、数据集：**是同一个文件，逐字节相同**

论文的 countdown 训练数据就是本仓库的 `countdown/data/countdown.json`——
`es_fine-tuning_countdown.py:192` 硬编码读这个路径，我们的
`axis_probe/src/data_countdown.py:12` 读的也是同一个绝对路径。**不存在"版本不同"的问题。**

数据集特征（实测）：

| 属性 | 值 |
|---|---|
| 行数 | 2200 |
| 字段 | `id`, `context`, `numbers`, `target`, `solution` |
| 3 数题 / 4 数题 | 1071 / 1129 |
| 操作数范围 | 1..99 |
| 目标值范围 | −2640 .. 1349 |
| 重复 (numbers, target) | 0 |
| prompt 形式 | 数据集自带 `context`，completion 式，以 `<think>` 结尾，**不套 chat template** |

论文正文把 countdown 归给 Gandhi et al. 2024 (Stream-of-Search) 和 Pan et al. (TinyZero)。
3-to-4 数、操作数 1..99 的形态与 TinyZero 的 `Countdown-Tasks-3to4` 一致。

### 数据质量问题：43 行（1.95%）目标值不是整数

```
id=204  numbers=[98, 47, 47, 52]  target='27.4'                 solution='((98 / (47 - 52)) + 47)'
id=254  numbers=[60, 10, 57, 80]  target='266.6666666666667'    solution='((10 / (60 - 57)) * 80)'
id=263  numbers=[99, 86, 76, 97]  target='-15.727272727272727'  solution='((97 + 76) / (86 - 97))'
```

生成器把除法结果当成目标值写了进去。prompt 会字面要求模型"create an equation that equals
266.6666666666667"。奖励函数 `answer_reward_function` 判定条件是
`abs(float(result) - float(target)) < 1e-5`，所以这些题**理论上可解但实际近乎不可能**。

分布：

| 切片 | 非整数目标 | 占比 |
|---|---|---|
| `[:300]`（我们的 eval） | 4 | 1.33% |
| `[:1000]`（论文的训练集） | 21 | 2.10% |
| `[300:]`（我们的训练池） | 39 | 2.05% |
| 全集 | 43 | 1.95% |

**影响**：我们 300 题 eval 的实际准确率天花板是 0.9867 而非 1.0。当前测量值在 0.01–0.12
区间，所以这**不影响任何现有结论**，纯粹是记录在案。达到 0.90+ 时才需要重新考虑。

## 二、论文的 countdown 超参（源：官方代码）

| 项 | 论文值 | 出处 |
|---|---|---|
| 模型 | Qwen2.5-**3B**-Instruct（默认）；表 1 共 7 个模型 | `:22` |
| 迭代数 | **500** | `:32` |
| 种群 N | **30** | `:33` |
| σ | **0.001** | `:34` |
| α | **0.0005** | `:35` |
| max_new_tokens | **1024** | `:36` |
| 采样 | **greedy**（`do_sample=False`） | `:37` |
| 精度 | bf16 | `:24` |
| 每成员评测题数 | **200**（附录 A.1）／代码默认 1000；**均为每代同一批** | A.1；`:211` `:153-158` |
| 训练集切片 | `data[:200]`（附录 A.1）／`data[:1000]`（代码默认），**固定，不重采样** | A.1；`:27` `:211` |
| 扰动方式 | **单边**：`+σε` → 评测 → `−σε` 还原，**非对偶** | `:147` `:172` |
| 更新 | `θ += α · (1/N) Σ zₙ εₙ`，`z = (r−mean)/(std+1e-8)` | `:337-352` |
| 每代噪声种子 | `np.random.randint(0, 2**30, size=30)`，每代重抽 | `:259` |
| reward | `0.1 · format_reward + answer_reward` | `countdown_task.py:78-95` |
| 总生成量 | 500 × 30 × 200 = **300 万次**（按 A.1）；按代码默认 1000 则为 1500 万 | 推算 |
| 评测集 | **另外 2000 题，与训练的 200 题分开** | A.1 |
| RL baseline 预算 | **无绝对数字**：GRPO/PPO 跑到耗完与 ES 相同的 sample-evaluation 总量即停 | A.1 |

`format_reward` ∈ {0, 0.1, 0.5, 0.6, 1.0}（`<think>` 匹配 +0.1，`<answer>` 匹配 +0.5，
完整格式直接 1.0），`answer_reward` ∈ {0, 1}。所以论文 reward 上限 1.1，且**在没人做对时仍有
格式分提供梯度信号**。

### 论文报告的结果（表 1，accuracy %）

| 模型 | Original | 最强 RL baseline | **ES** |
|---|---|---|---|
| Qwen-2.5-0.5B | 0.1 | 13.5 (Dr.GRPO-v) | 14.4 |
| **Qwen-2.5-1.5B** | **0.7** | **31.0 (Dr.GRPO-v)** | **37.3** |
| Qwen-2.5-3B | 10.0 | 43.8 (Dr.GRPO-v) | 60.5 |
| Qwen-2.5-7B | 31.2 | 57.5 | 66.8 |
| Llama-3.2-1B | 0.4 | 14.9 | 16.8 |
| Llama-3.2-3B | 3.2 | 47.8 | 51.6 |
| Llama-3.1-8B | 8.1 | 51.3 | 61.2 |

**我们这轮的对标值是 Qwen-2.5-1.5B 那一行：0.7% → 37.3%。**
我们实测 base = 1.00%（论文 0.7%，同量级），step 20 = 11.67%。

注：论文对 RL baseline 逐模型做了 β 和 α 的网格搜索，对 ES 只用一套固定超参——这个不对称
在解读 "ES 胜出" 时必须一并陈述。

## 三、我们的设置与论文的差异清单

| 项 | 论文 | 我们（本轮） | 差异性质 |
|---|---|---|---|
| 模型 | Qwen2.5-3B-Instruct（表 1 也有 1.5B） | Qwen2.5-**1.5B**-Instruct | 对标表 1 的 1.5B 行，✅ 可比 |
| N | 30 | 30 | ✅ 一致 |
| σ | 1e-3 | 1e-3 | ✅ 一致 |
| α | 5e-4 | 5e-4 | ✅ 一致 |
| 每成员题数 | 1000 | 1000 | ✅ 数量一致 |
| **迭代数** | **500** | **100** | ❌ **只有 1/5** |
| **批次策略** | **固定 `[:1000]`，目标函数平稳** | **每步从 `[300:]` 有放回重采样 1000** | ❌ 目标函数非平稳 |
| **训练/评测切分** | 训练 `[:1000]`，评测集未在代码中给出 | 训练 `[300:]`，评测 `[:300]`，**构造上不相交** | ⚠️ 我们更严格 |
| **reward** | `0.1·format + answer`（上限 1.1） | **纯二值 `answer_reward`** | ❌ 我们没有格式分 |
| max_tokens | 1024 | **2048** | ❌ 我们更宽 |
| 采样 | greedy | greedy (`temperature=0.0`) | ✅ 一致 |
| 扰动 | 单边 | 单边（vanilla 臂 `probe=False`，无对偶对） | ✅ 一致 |
| 精度 | bf16 | **float16** | ❌ 不同 |
| 推理后端 | HF `model.generate` | **vLLM** | ⚠️ 实现差异 |
| 噪声相关性 | 每个参数张量**同一个种子**（部分相关噪声） | 见下 | ⚠️ 待核 |

### 关于"部分相关噪声"

论文 README 自己承认：原始实现对每个参数张量用**同一个** `manual_seed(seed)`，噪声在张量之间
是相关的，不是 i.i.d.。作者另发布了 `es_fine-tuning_countdown_iid.py` 修正版
（`gen.manual_seed(seed + seed_shift)`，`seed_shift` 逐张量递增）。见
[作者的 discussion #7](https://github.com/VsonicV/es-fine-tuning-paper/discussions/7)。
**表 1 的数字来自哪个版本，论文正文未说明。**

## 四、结论与待解

**数据集完全相同**，不存在数据来源的混淆。真正的偏离集中在三处，按影响排序：

1. **迭代数 100 vs 500。** 这是最大的一处，也正是台账里 P2 那条"Countdown ES 跑满 500 步 —
   ES 是否只是欠训练"所指的分歧。按当前 53.3 min/step 实测速度，跑满 500 步需要
   **~444 GPU-h/臂（18.5 天）**，双臂 37 天——以现有硬件不可行。**这意味着我们无法直接复现
   论文的 37.3%，只能报告"在 1/5 预算下达到了 X"。** 这一点必须写进任何对外结论。
2. **reward 少了格式分。** 论文的 `0.1·format_reward` 在训练早期（无人做对时）仍提供可分辨的
   fitness；我们的纯二值奖励在那个阶段完全靠"有人偶然做对"。B=1000 下实测 ρ=0.81，说明这一点
   在当前配置下没有造成可分辨性问题，但它是一个未受控的差异。
3. **批次非平稳。** 论文优化一个固定的 1000 题目标函数；我们每步换一批。这让我们的 fitness
   曲线含有额外的批次噪声，且严格来说优化的不是同一个目标。

### 2026-08-23 更新：train-on-test 的疑问已解除（附录 A.1 读到了）

改用 ar5iv 镜像抓取，读到了此前被截断的附录 A.1。**论文在 held-out 集上报告**：
"200 samples for training, a separate 2000 samples for testing"，ES 的表 1 数字是训练 500 迭代
后在那 2000 题测试集上测的。所以**上一版列为"最重要未解项"的 train-on-test 风险不成立，
我们对标 1.5B 那行 0.7% → 37.3% 是合法的**。

2200 = 200 + 2000 正好用完整个数据集，所以论文的测试集应当就是 `[200:]`。我们用 `[:300]`
评测、`[300:]` 训练——切分点不同，但同样构造上不相交。

由此**总生成量从 1500 万更正为 300 万**：500 × 30 × 200。上一版按代码默认值 `--data_sample
1000` 推算，但 A.1 明写 200，且论文 README 的 countdown 示例命令用的也是 `--data_sample 200`
——两处一致，所以以 200 为准。这同时意味着**我们这轮的 B 比论文大**：论文每成员看 200 题固定
批，我们 1.5B 那轮用 B=1000、7B 这轮用 B=100（且每步重采样）。

**RL baseline 的预算：论文没给绝对数字。** A.1 有两句相对声明（2026-08-31 逐字复核 v2/v3
HTML，上一版此处的引文有误，写成了 "For all the ES, GRPO and PPO, ..."，实际原文如下）：

> For all the ES and RL baselines, the total number of sample evaluations was the same.

> For ES, results were reported on the test set after training for 500 iterations.
> For RL, the training was stopped after the same total number of sample evaluations as in the ES runs.

两句都在 **附录 A.1 Experimental Setup** 开头那段 "Experimental setup for the Countdown
experiments." 里，v2 与 v3 逐字相同。**计数单位是 "sample evaluations" 而不是 generations**，
该词全文只在 **A.8** 定义过一次："For each sample evaluation (processing one training data
point), ES needs to perform one forward pass on the base model, whereas GRPO needs to perform
one forward pass on the base model, one forward pass on the reference model, and one backward
pass (backpropagation) on the policy."

**遗留的歧义**：GRPO 一个 prompt 展开 group size N=8 条 rollout 时，算 1 次还是 8 次 sample
evaluation，论文从未言明。A.8 的 FLOPs 记账（ES 2PL vs GRPO 8PL，逐 data point 比）暗示按
"每条生成"计，即等价于 generation 对齐，但这是推断不是原文。全文不出现 "budget" /
"matched" / "equal number"，也不出现 generation 意义上的 "number of generations"。

**歧义可以被 A.8 自己的算式定死（2026-08-31 补）**。A.8 逐字写的是
"ES needs 2PL FLOPs for each sample evaluation, whereas GRPO needs 2+2+4=8PL"。把两种读法
代进这个算式：

- 读法甲（1 次 = 1 道题，GRPO 每题展开 G=8 条 rollout）：GRPO 需要 8 次生成前向 + 8 次
  reference 前向 + 对 8 条回答反向 = 8×(2+2+4) = **64PL**，与论文印的 8PL 差 8 倍。**不自洽。**
- 读法乙（1 次 = 1 条被生成并打分的回答）：GRPO 需要 1 次 base 前向 + 1 次 reference 前向
  + 1 次反向 = **2+2+4 = 8PL**，与论文逐字相符。**自洽。**

所以论文自己的记账只在读法乙下成立，而读法乙**恰好等价于生成次数对齐**——两法生成总数相同。
旁证：读法甲会让论文的 GRPO 吃到 2400 万次生成（7 个模型 × 约 5 种 RL 配置 × β/α 网格，
算力上不现实），且与本仓库 GRPO 用 30 万次生成得到 0.687 而论文 RL 用 2400 万次只有 0.438
（少 80 倍反而更好）严重冲突。**但论文正文从未言明，以上是推断。**

ES 侧的绝对量可由 A.1 反推：500 迭代 × N=30 × 200 题 = **300 万次** sample evaluation
（论文自己没印这个数）。RL 侧无任何绝对数字。**上一版说"batch size 在 A.1 中缺失"是错的**：
A.1 对 VERL 实现明确给了 "we set the global batch size of 1024, a learning rate of
1×10⁻⁶, and a rollout group size of N=8"；缺的是**步数与 epoch 数**。GRPO-z 另跑了两个
group size：N=8（countdown 的 common practice）与 N=30（对齐 ES 种群）。

**仍未确定**：表 1 用什么解码测（greedy 与否未在 A.1 说明）；正文指向的 "Table 4" 实际是
conciseness 任务的两条训练样例，β/α 网格在 Table 3，编号对不上，网格的具体取值范围未逐格读到。

## 本文的局限

- reward 公式、扰动方式、噪声种子等来自**官方代码**而非论文正文；若两者不一致，本文会跟着错。
  附录 A.1 已于 08-23 通过 ar5iv 读到（训练/测试切分、预算 parity），**A.3 仍未读到**。
- A.1 与代码在 `data_sample` 上不一致（200 vs 默认 1000）。本文以 A.1 的 200 为准，理由是 README
  示例命令同为 200；但**表 1 究竟用哪个跑的，论文没有说**。若实际是 1000，总生成量应为 1500 万。
- A.1 的内容因抓取工具的字符限制未能逐字全文复现，仅核到关键片段。
- 论文表 1 的数字是从 HTML 版抓取转录的，未逐格核对 PDF。
- 数据集统计（2200 行、43 行非整数目标等）是本地实测，可复现。
- 未跑任何 GPU 实验。
