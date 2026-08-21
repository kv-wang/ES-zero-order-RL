# Countdown 线接入八月 try:配置、移植与冒烟发现 (2026-08-15)

**状态:冒烟进行中(tmux `cdsmoke`)。** 本文记录配置、移植内容,以及冒烟已经查出的两个缺陷。
正式跑的结果另文报告。

## 为什么要接这条线

八月全部 null 结果的统一解释是 regime:数学任务上 split-half ρ 中位数精确为 0
(见 [BASE_MODEL_RESULTS](BASE_MODEL_RESULTS.md)),而 B=8 的奖励粒度是根因
(见 [PAPER_FIDELITY_AUDIT](PAPER_FIDELITY_AUDIT.md))。countdown 是本仓库**唯一**测到
ES 真有信号的 regime(07-30:ρ=0.51,base 0.08 → full-ES 0.450),而它恰好跑在
**N=30、B=100** 上——接近论文配置。

## 任务定义

**数据** `countdown/data/countdown.json`,2200 行。给几个数,用 `+ - * /` 各数恰好用一次
凑出目标数。

```
numbers   [44, 19, 35]     target  98     solution  ((44 + 19) + 35)
context   数据集自带的完整提示词,结尾是未闭合的 <think>,补全式,不套 chat template
```

**切分** 行 `[:300]` 评测 / 行 `[300:]` 训练(1900 行)。构造上不相交,无需去污染。

**奖励** `countdown_task.answer_reward_function`,二元:取最后一个 `<answer>`;内容只允许
`[0-9+-*/() ]`;抽出的数字排序后必须与 numbers 完全相同;求值等于 target 得 1.0,否则 0.0。

ES 训练器、verl 的 GRPO 奖励、`eval_core._eval_countdown` **共用这一个函数**。
该路径**完全不经过 `ood_eval/answer_extraction`**,所以 08-09 的 `\boxed{}` 缺陷从未污染过
countdown——这正是 07-30 的结果至今有效而七月全部数学数字作废的原因。

**评测** 贪心,`max_tokens=2048`(训练与评测同一上限,07-30 协议明确"绝不混用"),
cap 300。countdown 为 ID,六套数学集为 OOD——方向与数学线相反。

## 移植内容(八月 try 此前完全没有 countdown 支持)

| 组件 | 做法 |
|---|---|
| `es_bench/data_countdown.py` | 从 07-24 track_b 拷入 |
| `es_train_axis.py` | 加 `--dataset countdown` 与 `--batch/--sigma/--alpha/--max_tokens` |
| `eval_core.py` | 移植 `_eval_countdown` + `include_countdown` 开关 |
| `eval_base.py` | 加 `--include_countdown` |
| `run_grpo.sh` | 加 `TASK=countdown`(切数据/奖励/恒等模板),**不另写第二个 GRPO 脚本** |
| `run_countdown_all_arms.sh` | 新建驱动:base → GRPO → vanilla → baseaxis → momentum → 读出 |

GRPO 最易错的一环:countdown 的提示词是原始 `context`、不是对话轮次,而 verl 一定会套
chat template。因此训练用**恒等模板的模型副本**,合并成 HF 后再把真 tokenizer 还原回去
——否则后续数学 OOD 评测会用一个剥掉 chat 标记的模板。

## 冒烟发现的两个缺陷

**一、`shutil` 未导入(我 08-14 引入)。** 给 `eval_grpo.py` 加 tokenizer 还原时,替换目标
写成独立子串 `import subprocess`,而实际是合并导入行,未匹配;断言只检查"shutil 不存在",
静默通过。已修,并加静态检查确认所用模块均已导入。

**二、失败被记成成功(测量有效性缺陷,更严重)。**

```bash
echo "[$(date +%F_%T)] STAGE3 rc=$? " | tee -a "$DRIVER"
```

`$(date)` 先执行并把 `$?` 重置为 0,之后 `$?` 才展开。**每一次 stage 3 失败都会被记为 rc=0**,
日志同时打印 `COMPLETE -> ...summary.json`,而该文件并不存在。

已修 `run_grpo.sh` 与 `run_countdown_all_arms.sh` 两处:先捕获 rc,失败时打印日志路径并
以非零码退出。

**影响范围核查:此前三次 GRPO 运行(math gen400 / math manual20 / base gen400)的 summary
均存在,没有历史结果被静默吞掉。** 该缺陷只造成本次冒烟的误报。

## 另两个静默失效(2026-08-15,正式跑启动前发现)

**三、`ES_GENERATIONS` 硬编码为数学线的形状。** `run_grpo.sh` 里写死
`ES_GENERATIONS=$((200*16*8))` = 25,600。`BUDGET=gen` 本应让 GRPO 对齐 ES 的总生成数,
但 countdown 的 ES 臂是 100×30×100 = **300,000**。照原样跑,GRPO 只得 200 步而非 2343 步
——**少 12 倍,且不报错**。已改为可覆盖,由 countdown 驱动按实际形状传入。

**四、基线跨 cap 复用。** `stage_base` 见到 `base_summary.json` 即跳过,而冒烟的基线是
cap=40。正式跑(cap=300)会直接复用那份 40 题的基线,使每个臂都在跟一个 40 题参照比。
已把 cap 写进目录名(`countdown_base_<model>_cap<N>`)。

这两个与前两个同类:**脚本会静默给出错误配置或错误结论,不会报错**。四个里有三个
(rc 掩盖、ES_GENERATIONS、cap 复用)都属于"沉默地做错事",只有 `shutil` 那个会崩。

## 未训练基线(正式,cap=300)

| | countdown(ID) | math500 | gsm8k | svamp | minerva | olympiad | amc23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| n | 300 | 300 | 300 | 300 | 272 | 300 | 40 |
| acc | **0.010** | 0.5133 | 0.7500 | 0.8167 | 0.0846 | 0.1500 | 0.4000 |
| extract | **0.7533** | 0.970 | 0.980 | 0.9633 | 0.9669 | 0.900 | 1.000 |

**countdown 起点 0.010**(300 题答对 3 题),比 07-30 的 3B-Instruct(0.08)还低一个量级,
落在论文全部实验 0.1%–31.2% 区间的下沿。提升空间几乎是全部,不存在数学线上
base 已达 0.65 的"窄靶"问题——这正是要找的 regime。

**注意:冒烟阶段(cap=40)测得的 0.025 已被本表取代。** 那是 40 题里答对 1 题,属噪声;
不要引用。这也是上面第四个缺陷必须修的原因。

**但 countdown 的 extract_rate 只有 0.725** —— 这里"抽取成功"指输出同时含
`<answer>` 与 `</answer>`,即 27.5% 的回答根本没按格式作答,明显低于数学集的 0.95+。
因此 0.025 里混着"不会做"与"没按格式作答"两种情形。若训练后 extract_rate 上升,
则提升中有一部分是**格式学习**而非推理能力,报告时必须分开陈述。

## 未解决

**冒烟未完成。** ES vanilla / baseaxis 的 countdown 路径(原始 `context` 提示词)尚未验证通过;
GRPO 臂需在修复后重跑 stage 1(checkpoint 仍在,只需重做合并+评测)。

**成本未实测。** 07-30 在 3B 上同形状(100 步 × N=30 × B=100 = 30 万次生成)为 8.63 h/臂;
1.5B 的实际耗时要等正式跑才知道。

**预算不对称。** 一个 ES 臂 30 万次生成,GRPO 按 128 rollouts/step 需 2343 步才能对齐;
07-30 用的步数对齐只给 GRPO 1.28 万次生成(23 倍劣势),其报告已自行标注这会低估 GRPO。
新脚本默认 `BUDGET=gen`,但该模式尚未实跑验证过耗时。

**单 seed。** 脚本默认单 seed,低于本项目 3-seed 预注册规则,**不足以支撑"哪个变体更好"**,
只能回答"在有信号的 regime 里 ES 变体是否移动"。
