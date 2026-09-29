# GRPO 周期评测：内存权重 + 末步单 checkpoint（2026-08-24）

## 问题

能不能让 GRPO 的评测直接用**训练过程中的内存权重**，不落 checkpoint，训练结束自动评一次，
并支持每隔若干步评一次？

动机：7B 的 FSDP 分片单份约 84 GB（model 28 + Adam 状态约 56；此前记的"90 GB"是估值），
正在跑的 `gen2344` 按 `save_freq=200` 会写 11 份约 0.9 TB，
而其中只有最后一份会被用来评测。原链路是
`训练 → 存 FSDP 分片 → 合并成 HF → 单卡起 vLLM 重评`，三步里有两步纯粹是为了把权重搬出进程。

**答案：可以，用 verl 自带的 `test_freq` + `val_before_train`，stage 3 退化为零 GPU 的日志解析。**
6 步冒烟（`EVAL_EVERY=3`）实测三次评测全部落盘，无任何 checkpoint 目录被创建。

**2026-08-24 追加（按用户指示）**：OOD 互斥已解决——改为**保留末步一份 checkpoint**，
`SAVE_EVERY=last`（默认）令 `save_freq=$STEPS`，`save_contents=[model]` 只存 model 分片。
于是两条评测**同时存在**：训练中的内存曲线（countdown）+ 末步合并后的六套电池（+KL +extract_rate）。
2 步冒烟全链路 rc=0，`global_step_*` 目录**恰好一个**（43 GiB）。
**顺带修掉一个使 GRPO 的 KL 列一直无意义的缺陷**，见"KL 缺陷"一节。

## 设置

改 [`run_grpo_7b_countdown.sh`](es_bench/axis_probe/run_grpo_7b_countdown.sh) 的 stage 2/3，
新增 [`grpo/collect_val_curve.py`](es_bench/axis_probe/grpo/collect_val_curve.py)。

| 项 | 值 |
|---|---|
| 模型 / 任务 | `Qwen2.5-7B-Instruct`，countdown |
| 冒烟规模 | `BUDGET=manual STEPS=6`，`EVAL_EVERY=3` |
| `trainer.save_freq` | **-1（不写任何 checkpoint）** |
| `trainer.test_freq` | `$EVAL_EVERY`，默认 100（**bs=64 → 586 步 → 7 次**，含 step 0）；`-1` 则只在训练末评一次 |
| `trainer.val_before_train` | **True**（step 0 作本链路自己的 base 参照） |
| 评测集 | `data.val_files` = countdown `val.parquet`，300 题（= ES 臂的 `[:300]` 同一切片） |
| 解码 | verl val 默认贪心、n=1 |
| 奖励 | 同一个 `reward_countdown.py` → 同一个 `countdown_task.answer_reward_function` |
| 其余训练参数 | TRAIN_BS=16、GROUP=8、MICRO=2、TP=2、util=0.5、max_prompt/resp=1024/2048 |

countdown 这一列与 ES 臂**仍然可配对**：题目切片、解码、奖励判定三项同源。

## 结果：三次评测全部产出，零 checkpoint

| step | countdown accuracy | 命中键 |
|---|---|---|
| 0（val_before_train） | **0.3267** | `countdown_es/acc/mean@1` |
| 3 | **0.3633** | 同上 |
| 6（训练末自动） | **0.3700** | 同上 |

`/home/hyin66/es_ckpts/grpo_countdown_7b_bs16g8_manual6/` **不存在**（`SAVE_EVERY=-1` 下驱动不
mkdir，verl 也没写）；正在跑的 `gen2344/` 目录 4.0K 即空壳。stage 3 零 GPU、秒级完成。

6 步不足以谈学习趋势，这三个数的用途是证明链路通，不是证明 GRPO 在学。

## 仪器发现：verl 0.6 把 countdown 指标拆到两个前缀

stage 3 第一次跑**直接崩**：`ValueError: max() arg is an empty sequence`。根因是我按
`val-core/{ds}/reward/mean@1` 抽数，而 verl 0.6 实际写的是：

- `val-core/countdown_es/acc/mean@1`
- `val-aux/countdown_es/reward/mean@1`

两者**数值恒等**（0.3267 / 0.3633 / 0.3700 三点全等），因为 `reward_countdown.py` 是二值。
现已改为优先读 `acc`、回退 `reward`。

**同一处修出第二个 bug**：行筛选只认 `val-core/`，而回退键只存在于 `val-aux/` 下 →
`reward` 回退分支**永远走不到**。当前 verl 上两个前缀总是同行出现，故正式跑不受影响，
但该分支对旧日志无效。已改为两个前缀都接。

**加固**：evalcurve 每行记 `acc_key`，summary 记 `acc_keys` 与 `acc_key_expected`。
verl 若再改键名，兜底可能捡到无关的 `/mean` 键并给出一个看着像准确率的数——
合成日志实测确认该情形下 `acc_key_expected=False`（键 `score/mean@1`，值 0.41 仍可见）。
这是"宁可给可见的错数，不要静默丢弃"的取舍。

回归 5 组全过：两前缀同行 / 仅 val-aux 走回退 / 键改名可见但被标记 / 无 val 行 rc=2 /
有 val 行但无可用指标 rc=2。

## 末步 checkpoint：设置与 2 步冒烟

`SAVE_EVERY=last`（新默认）→ `trainer.save_freq=$STEPS`。verl 在 `global_steps % save_freq == 0`
时存盘、`global_steps` 走 1..STEPS，故**只在末步命中一次**。另加 `max_actor_ckpt_to_keep=1` 兜底。

**两处不得不改的现实**：

| 项 | 原计划 | 实际 | 原因 |
|---|---|---|---|
| 落盘位置 | `/home/hyin66/es_ckpts` | `/var/tmp/es_ckpts`（`CKPT_ROOT` 可覆盖） | 家目录只剩 **13 GiB**，装不下；overlay 有 175 GiB |
| 存盘内容 | 默认（model+optimizer+extra） | **`save_contents=[model]`** | 默认约 84 GB（Adam 状态占两倍参数量）；只留 model 为 28 GB fp32 分片 |

代价：**该 checkpoint 不可用于续训**（无 optimizer 状态），只够合并成 HF 做评测。
`huggingface/`（config+tokenizer）无论 `save_contents` 如何都会写，故合并器所需文件齐全。

2 步冒烟（`BUDGET=manual STEPS=2 EVAL_EVERY=2`，TRAIN_BS=8）全链路 rc=0：

| 阶段 | 结果 |
|---|---|
| STAGE2 训练 | rc=0，460 s（230.0 s/step，含引擎启动） |
| `global_step_*` 目录数 | **恰好 1 个**（`global_step_2`，**43 GiB** 实测，比预估 28 GB 高） |
| STAGE3a 内存曲线 | rc=0，2 次评测：step 0 **0.3167** → step 2 **0.3700**；`acc_key_expected=True` |
| STAGE3b-pre base KL 捕获 | rc=0，62 s |
| STAGE3b 合并 + 六套电池 | rc=0，194 s |

电池实测（`EVAL_CAP=40`，冒烟档）：math500 0.675、gsm8k 0.875、svamp 0.925、
minerva 0.175、olympiad 0.125、amc23 0.550、countdown 0.375（extract **0.675**）。
**这些数字只证明链路通**：2 步不可能训出什么，且 cap=40 的标准误约 ±0.077。

## 正式配置切到 TRAIN_BS=64（2026-08-24，按用户指示）

驱动默认值改为 `TRAIN_BS=64`、`EVAL_EVERY=100`、`SAVE_EVERY=last`。`GROUP=8` 不变，
故 512 rollouts/step，生成次数对齐 300,000 → `BUDGET=gen` 得 **586 步**（300,032，超 0.01%）。

| 项 | bs=16（旧默认） | **bs=64（新默认）** |
|---|---|---|
| rollouts/step | 128 | **512** |
| 对齐 300k 所需步数 | 2,344 | **586** |
| countdown 评测次数（`EVAL_EVERY=100`） | 25 | **7**（step 0/100/…/500/586） |
| 实测 s/step | ~100 | **393.2** |
| 外推总时长 | ~65 h | **~64 h** |

**墙钟基本不变**是应当的：生成次数已对齐，bs 只改变每步摊到多少次生成。
bs 换来的是**更少的权重更新次数**（586 vs 2,344）——08-13 预算审计里那条对 GRPO 有利的
不对称（更新次数 2,344 vs ES 的 100）因此从 23.4× 收窄到 **5.86×**，方向上更接近 ES 臂。

**同时修正评测次数的算法**：原 `N_EVALS` 漏算 `val_before_train` 的 step 0，586 步下报 6 而实为 7。

### 2 步冒烟（bs=64，`EVAL_CAP=40`）

全链路 rc=0。`global_step_*` **恰好 1 个**（43 GiB，与 bs=8 那次同值——分片大小由参数量定，与 bs/步数无关）。

| 项 | 实测 |
|---|---|
| 整卡峰值 | **122.42 / 122.84 GiB/卡**，双卡 245.27 GiB |
| s/step | 372.8（step 1，含引擎启动与 step-0 评测）→ 396.9（step 2），均值 393.2 |
| 内存曲线 | step 0 **0.3033** → step 2 **0.3300**；`acc_key_expected=True` |
| KL | **0.1684**，地板 7.193e-06 的 **23,414 倍** → 走的是修复后的 base 参照路径 |
| 电池（cap=40） | math500 0.675、gsm8k 0.875、svamp 0.925、minerva 0.175、olympiad 0.125、amc23 0.550、countdown 0.425（extract 0.675） |

峰值与 08-24 扫描在同参数下测的 122.82 GiB/卡吻合到 0.05 GiB，且**那轮没有 val 也没有存盘**，
本轮三者同时发生仍未 OOM（143,771 MiB/卡，余量约 21 GiB）→ bs=64 + 周期评测 + 末步存盘可共存。

**这些数字不产生任何准确率结论**：2 步训不出东西，cap=40 的标准误约 ±0.077，
`0.3033 → 0.3300` 的 +0.027 落在 300 题的 ±0.026 边缘且 2 步内不该有可测进展。
冒烟那份 43 GiB 已删（overlay 回到 175 GiB）。

## KL 缺陷：GRPO 的 `kl_proxy_drift` 一直在拿模型和自己比

改动中发现的，与 checkpoint 无关，但波及已发表的数字。

`kl.drift(rec)` 的定义要求 `rec` 在**未训练的 base 权重**上捕获：
D = mean_t ( logP_base(y_t) − logP_theta(y_t) )，y 是 base 的贪心输出。
ES 臂天然满足——[es_train_axis.py:206](es_bench/axis_probe/src/es_train_axis.py#L206) 在 step 0
更新前调 `capture_base`，末步才调 `drift`。但 `eval_grpo.py:75-78` 进程里**只有合并出的训练后模型**，
两个调用都打在它身上 → 记录的是训练后模型对自己贪心输出的 logprob，`drift` 再拿它和自己比。

新增 [`src/kl_capture.py`](es_bench/axis_probe/src/kl_capture.py) 作独立进程（两个 vLLM 引擎不能
同进程共存），先在 base 上捕获并存 json，`eval_grpo.py --kl_base_rec` 读入；记录钉死
model/max_tokens/KL_N_PROMPTS/DATA_SEED/LEVELS/TRAIN_SIZE，模型不符则拒绝。

**同权重自漂移 = 该估计量的数值地板，实测 7.193e-06。** 这一个数就足以判定旧数字：

| 数字 | 值 | 与地板的关系 |
|---|---|---|
| 08-13 GRPO 报告的 `kl_proxy_drift` | −2.11e-7 | **比地板低 34 倍**（且为负）→ 只能是自比 |
| 本轮修复后（2 步 GRPO） | **0.168** | 高出地板 **4 个数量级** → 真实可测 |

**故 08-13 那条「GRPO 的 KL=−2.11e-7，权重几乎未动，疑为 FSDP→HF 合并舍入」作废**：它量的是
自一致性地板，不是位移。当轮由此推出的「ES 走得远但方向随机 / GRPO 走得近但方向准确」
**在 KL 这一半上没有证据支撑**（ES 侧的 KL 是对的，GRPO 侧的不是），台账 08-13 两行涉及
GRPO KL 的表述需按此重读。**未做**：没有回填旧运行的 KL——`gen2344` 已被 kill 且从未写出
checkpoint，08-13 的合并权重也不在了，只能在下一次正式跑上测。

## 局限

- **单 seed**（seed 概念在本轮不适用：6 步冒烟只跑一次，无运行间方差估计）。三个数字的
  运行间抖动未知，`0.3267 → 0.3700` 的上升**不可读作学习**——300 题的标准误约 ±0.026，
  这个 +0.043 只勉强出边，且 6 步内 GRPO 不该有可测进展。
- **step 0 的 0.3267 与 08-23 独立测的 7B base 0.3000 差 2.7 个百分点。** 两条评测链不完全
  同源：本链路的 prompt 走 `val.parquet`，08-23 走 `eval_base.py` 的构造路径。题目切片、
  解码、奖励判定三项相同，差的是 prompt 组装。**故本链路的 step 0 只能作本链路内部的参照，
  不可与 08-23 的 base 数字并列**，也不可用来声称"prompt 链与论文一致"（那条结论属 08-23）。
- **OOD 互斥已按用户指示解决（保留末步一份），但代价是实打实的**：43 GiB 落在**易失的
  overlay** 上（`/var/tmp`，非家目录），且 `save_contents=[model]` 使其**不可续训**。
  即 checkpoint 只是"为了评测把权重搬出进程"的中转，不提供任何抗中断能力。
- **两条 countdown 数字不是同一个测量，容易被并列引用**：内存曲线用 live 权重 + `val.parquet`
  提示词，电池用合并权重 + `eval_core` 提示词，step 0 实测差约 2.7 个百分点。驱动的收尾输出
  已显式印出这句警告，但**报告里引用时仍须说明用的是哪一条**。
- **extract_rate 与六套 OOD 只在末步有一个点**，中间步只有 countdown 一列 → 训练过程中
  "格式遵从 vs 推理能力"的**拆分仍然只能在末步做**，拿不到曲线。而 7B base 的 countdown
  extract_rate 仅 0.620，vanilla 臂那 +26.7 点是两条路径都动，故中间步与 ES 臂比较时这项不可比。
- **每次评测约 100 s**（300 题贪心 2048 token），与一个训练步同量级。`EVAL_EVERY=100` 时
  约 24 × 100 s = 40 min，占 65 h 的 1%；`EVAL_EVERY` 调小到 10 则涨到约 6.5 h（10%）。
- **中断仍然无产物可续**，且这条**没有被末步 checkpoint 改善**：末步存盘意味着中途中断时
  什么也没写；`save_contents=[model]` 又去掉了 optimizer 状态，故即使存了也不能续。
  08-23 的 B=1000 与 08-24 的 baseaxis 都是 tmux server 消失致死，正式跑必须放进 tmux。
- 冒烟分别只 6 步、2 步与 2 步（bs=64），未验证长跑下 `test_freq` 的累计开销、日志体积，以及
  7 次评测是否都能被解析；**43 GiB 在 bs=8 与 bs=64 两次都是 STEPS=2 的实测**，586 步的分片
  大小与步数无关（FSDP 分片只存参数）但未直接验证。
- **bs=64 的 393.2 s/step 取自 2 步**，其中 step 1 含引擎启动与 step-0 评测。`~64 h` 是按此
  外推 586 步，假设后 584 步同速；08-23/08-24 两次实测都显示外推可偏 1.65 倍（对 ES 臂是偏高），
  故该时长只作规划用。
- **bs=16→64 只改了每步摊到的生成次数与更新次数，不改任何已发表结论**：两档都不曾跑出准确率。
  更新次数从 2,344 降到 586 使 GRPO-vs-ES 的更新次数不对称收窄到 5.86×，但**仍未对齐**，
  且题目抽取次数与训练温度两项照旧不对齐，三项依然都对 GRPO 有利。
- **KL 地板 7.193e-06 是单次测量**，未估其运行间抖动；它足以判定 −2.11e-7 属自比
  （差 34 倍），但不宜当作精确阈值使用。

## 未解决

- **旧的 GRPO KL 数字无法回填**：`gen2344` 已被 kill 且从未写出 checkpoint（`save_freq=200`，
  step 200 未到达），08-13 的合并权重也已不存在 → 修复后的 KL 只能在下一次正式跑上取得。
- **7B GRPO 的准确率对照仍不存在**；本轮两次冒烟（6 步 / 2 步）都不产生准确率结论。
- 长跑下的解析可靠性未验证（见局限）。
- **overlay 上的 43 GiB 需要人工清理**（本轮冒烟的那份已删）；正式跑结束后同样要删。
- 三处与论文的已知偏离（迭代数、每步重采样 vs 固定批、reward 缺 `0.1·format`）不受本改动影响，
  仍未分离。

