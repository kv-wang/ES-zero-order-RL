# Qwen2.5-7B-Instruct GRPO 显存扫描：峰值 vs training batch size（2026-08-24）

## 问题

对 `Qwen2.5-7B-Instruct` 做 GRPO、**GROUP=8 固定**、双卡并行，`data.train_batch_size` 取不同值时
整卡显存峰值分别是多少？

动机：7B countdown 线目前只有 base/vanilla/baseaxis 三个 ES 臂，没有 GRPO 对照（`run_countdown_7b.sh`
没有 GRPO stage）。要补 GRPO 对照，先得知道 batch 能开到多大而不 OOM。

**答案（bs ∈ {8,16,32,64}，各 5 步，全部 rc=0）：整卡峰值 122.82–122.90 GiB/卡，
与 batch size 无关**，8 倍的 batch 差异只带来 0.08 GiB 的波动（在测量噪声内）。
torch 侧同样恒定（alloc 73.02 GB、reserved 112.12 GB）。原因是
`ppo_micro_batch_size_per_gpu=2` 固定——反向按 micro-batch 分块累积，bs 变大只是顺序多跑
几个 micro-batch。**bs 买的是时间（532 → 2093 s），不是显存。**

## 设置

新增 `axis_probe/probe_grpo_7b_bs.sh`。

| 项 | 值 |
|---|---|
| 模型 | `Qwen2.5-7B-Instruct`（15 GB 快照，bf16） |
| 任务 / 数据 | countdown，`grpo/data_countdown/train.parquet`（1900 行） |
| GROUP（`rollout.n`） | **8，固定** |
| 扫描量 | `data.train_batch_size` ∈ {4, 8, 16, 32, 64} |
| 步数 | 5（bs=4 那次是 2 步的冒烟，作参照行） |
| 卡 | 2×，`n_gpus_per_node=2`，FSDP actor |
| rollout | vLLM，TP=2，`gpu_memory_utilization=0.5` |
| MICRO（per-GPU） | 2（7B 上沿用数学线的 8 会 OOM） |
| max_prompt / max_resp | 1024 / 2048（= 7B ES 臂的 max_tokens） |
| 其他 | lr 1e-6、`kl_loss_coef=0.001`、`ppo_mini_batch_size` = train_batch_size、`save_freq=-1`、`val_before_train=False` |
| 采样 | `nvidia-smi` 每 1 s 采两卡 `memory.used` |
| 卡容量 | 143,771 MiB/卡 |

与 08-16 的 1.5B countdown GRPO 一致：数据、奖励、恒等 chat template、GROUP=8、lr、kl_coef。
不同：模型 7B、双卡（那轮单卡）、MICRO=2、MAXRESP=2048、只跑 5 步（那轮 2343 步）。
**本文只测显存，不产生任何准确率结论。**

### 为什么用 nvidia-smi 而不是 torch 计数器

2026-08-13 MEMORY_REPORT 的教训在本轮被再次验证：GRPO 的 vLLM 跑在独立进程
（`vLLMHttpServer`），而 verl 的 actor 指标在 TaskRunner 侧读，
`torch.cuda.max_memory_allocated()` 按进程统计，**看不见 KV cache**。bs=4 的对照：

| 量 | 值 |
|---|---|
| torch `max_memory_allocated` | 73.02 GiB |
| torch `max_memory_reserved`（step 1 → step 2） | 90.96 → 111.26 GiB |
| **nvidia-smi 整卡峰值** | **122.56 GiB/卡** |

差额就是 vLLM 的 KV。唯一有效测法是进程外的整卡采样。

## 结果：四档全部完成（5 步，rc=0，无一 OOM）

`results/grpo_mem_7b_20260824_102424/summary.jsonl`。bs=4 行来自另一次 2 步冒烟
（`..._101046/`），仅作参照，不与 5 步四档同列比较耗时。

| bs | rollouts/步 | 步数 | rc | 整卡峰值/卡 | 双卡合计 | torch alloc | torch reserved | 耗时 |
|---|---|---|---|---|---|---|---|---|
| 4（参照） | 32 | 2 | 0 | 122.56 GiB | 244.98 GiB | 73.02 GB | 111.26 GB | 345 s |
| 8 | 64 | 5 | 0 | **122.89 GiB** | 245.31 GiB | 73.02 GB | 112.12 GB | 532 s |
| 16 | 128 | 5 | 0 | **122.90 GiB** | 245.32 GiB | 73.02 GB | 112.12 GB | 747 s |
| 32 | 256 | 5 | 0 | **122.86 GiB** | 245.28 GiB | 73.02 GB | 112.12 GB | 1225 s |
| 64 | 512 | 5 | 0 | **122.82 GiB** | 245.25 GiB | 73.02 GB | 112.12 GB | 2093 s |

**峰值与 train_batch_size 无关。** 四档整卡峰值全部落在 122.82–122.90 GiB/卡 的
**0.08 GiB 带内**（相对幅度 0.07%），而 batch 翻了 8 倍（rollouts/步 64 → 512）。
带内的排序甚至是**微弱递减**的（bs=64 最低），即观测到的差异是 vLLM 分配的运行间抖动
（08-23 base 臂实测约 2 GB 量级），不是 batch 的效应。

**这不只是"固定 KV 压平了整卡曲线"——actor 侧本身也是平的。** torch 侧四档同样收敛：
`max_memory_allocated` 恒为 **73.02 GB**（差异在小数第 4 位，约 1 MB），
`max_memory_reserved` 恒为 **112.12 GB**（bs=4 那次只跑 2 步，尚未爬到该稳态，见下）。

机制是 **`ppo_micro_batch_size_per_gpu=2` 固定**：反向按 micro-batch 分块、梯度累积，
train_batch_size 变大只是**顺序多跑几个 micro-batch**，任一时刻的激活量由 MICRO 决定而非 bs。
耗时随 bs 近似线性增长（532 / 747 / 1225 / 2093 s）正是这个形态的另一面——
**bs 买的是时间，不是显存**。故本轮的实际结论是：显存的控制变量是 MICRO、max_resp 与
`gpu_memory_utilization`，不是 train_batch_size。

`max_memory_reserved` 的逐步爬升在四档完全一致：90.96 → 111.26 → 112.12 GB 后持平
（bs=4 只有 2 步，故停在 111.26，这解释了它整卡峰值 122.56 略低于其余三档的 0.3 GiB）。
即前 3 步是 PyTorch caching allocator 的暖机，**2 步的冒烟不足以测到稳态峰值**。

### 各档 `response_length/clip_ratio`（逐步）

| bs | 逐步 clip_ratio | 均值 |
|---|---|---|
| 4 | 0.4375, 0.3438 | 0.391 |
| 8 | 0.281, 0.266, 0.156, 0.328, 0.297 | 0.266 |
| 16 | 0.266, 0.211, 0.273, 0.320, 0.234 | 0.261 |
| 32 | 0.309, 0.344, 0.234, 0.270, 0.258 | 0.283 |
| 64 | 0.271, 0.238, 0.283, 0.248, 0.256 | 0.259 |

四档 25%–28% 的回答撞到 2048 上限，量级相当（bs=4 的 0.391 偏高是 2 步的小样本）。
故各档的长度饱和程度可比，峰值的可比性不受此项干扰。

## 结果：bs=4 参照点（2 步，rc=0）

| 项 | 值 |
|---|---|
| 峰值/卡 | **122.56 GiB**（GPU0 122.42 / GPU1 122.56） |
| 双卡合计峰值 | 244.98 GiB |
| rollouts/step | 32（= 4 × 8） |
| 耗时 | 345 s（含约 170 s 引擎启动） |
| 采样点 | 650 |
| `response_length/clip_ratio` | 0.4375 |

产物：`results/grpo_mem_7b_20260824_101046/summary.jsonl`、`bs4_g8_mem.csv`。

### 峰值出现在 update_actor，不在 rollout

双卡合计的时间线（相对启动）：

| t | 双卡合计 | 阶段 |
|---|---|---|
| 106 s | 28.91 GiB | 权重加载 |
| 212 s | 50.51 GiB | 引擎就绪 |
| 233 s | 164.04 GiB | rollout（生成） |
| 275 s | 51.71 GiB | 权重卸载/回收 |
| 296 s | 113.02 GiB | log_prob / ref |
| **339 s** | **244.98 GiB** | **update_actor（反向）** |

峰值由反向传播的激活 + 梯度 + 优化器状态决定，rollout 只是次高点（164 GiB）。这与 ES 的形态
相反——ES 没有反向，峰值完全由 KV 预算主导（08-13 已验证 ES 峰值随 util 线性移动）。

## 解读陷阱：固定 KV 预留，以及一处需要更正的预判

`rollout.gpu_memory_utilization=0.5` 按**卡容量的固定比例**预留 KV，与 batch size 无关：
143,771 × 0.5 ≈ 71.9 GiB/卡 被无条件占掉。所以整卡曲线是「常量 KV 预留 + 训练侧」之和。

**更正**：bs=4 那次我据此预判"batch 敏感性被这块常量压平了，要看真实敏感性须改读 torch
`max_memory_reserved`"。四档跑完后这条只对了一半——torch 侧**也是平的**
（alloc 恒 73.02 GB、reserved 恒 112.12 GB）。**真实的 batch 敏感性本来就不存在**，
因为 `ppo_micro_batch_size_per_gpu=2` 固定，反向按 micro-batch 分块累积。
固定 KV 确实会掩盖训练侧的变化，但本配置下训练侧并没有变化可供掩盖。

这也把"要分离 batch 敏感性还需另跑 util 扫描"这条未解决项作废：要移动峰值应扫 MICRO，
不是扫 bs 或 util。

## 局限

- **每档单次运行，无运行间方差估计。** 这对本文的核心结论是**实质限制**：四档差异
  0.08 GiB 落在 vLLM 分配抖动（08-23 base 臂实测约 2 GB）之内，所以"峰值与 bs 无关"
  严格说是"若存在 bs 效应，其幅度小于本测量的噪声底"。要给出真实的上界需重复测量估方差。
- **每档只跑 5 步。** `max_memory_reserved` 在第 3 步才到稳态（90.96 → 111.26 → 112.12 GB），
  故 5 步刚够；bs=4 的 2 步**没到稳态**，其整卡 122.56 GiB 因此偏低约 0.3 GiB，
  不可与其余四档等价比较。更长训练是否还有缓慢爬升未测。
- **只测了 MICRO=2 一个点。** 结论"峰值由 MICRO 而非 bs 决定"是从「bs 变而峰值不变」+
  「MICRO 决定分块大小」这一机制推断的，**未通过扫 MICRO 直接验证**。
- **`clip_ratio` 约 0.26**：四档都有约 26% 的回答撞到 2048 上限，KV 与激活吃满长度，
  故峰值是**该 max_resp 下的上界**，不是平均态。max_resp 改小峰值会降。
- **耗时含引擎启动**（约 170 s），不可直接读作 s/step；跨档比耗时时该常量未扣除。
- **固定 KV 预留（71.9 GiB/卡）占了整卡峰值的一半以上**，故整卡数字主要回答"会不会 OOM"。
  改 util 会整体平移全部四档。
- 与 08-16 的 1.5B GRPO 不可配对（模型、卡数、MICRO、max_resp 四项同时不同）。
- **只测显存，不产生任何准确率结论**；本文不能用于判断哪个 bs 训得更好。

## 未解决

- **MICRO 扫描未做**：这是本轮识别出的真正控制变量，但未实测。要知道 7B 双卡能开到多大的
  MICRO（以及能否用更大 MICRO 换速度），需要单独一轮扫描。
- **OOM 边界未触到**：四档 rc=0，bs 这条轴上没有边界可报；边界应在 MICRO 或 max_resp 上找。
- **运行间方差未估**，见上；0.08 GiB 的档间差异目前不可归因。
- 7B GRPO 的准确率对照仍不存在；本文不涉及。

## 附带发现：baseaxis 臂已中止于 step 49（非跑满）

查显存空闲时发现。**与 B=1000 那轮同一死法**：tmux server 消失，`KeyboardInterrupt` 打在
`shm_broadcast.acquire_read` 的自旋上，非 OOM、非 GPU 错误。

| 项 | 值 |
|---|---|
| 停止 | 08-24 09:46:23 最后写入 |
| 完成步数 | 49（jsonl 50 行 = steps 0–49），目标 100 |
| 已耗 | 约 9.15 h |
| 后续 | 驱动照常跑了 stage 4 readout 并打出 `ALL DONE` |

**含义**：stage 4 那份 REPORT.md 里的 baseaxis 列**只有 50 步，不是跑满的 100 步终值**，
不可与 vanilla 的 100 步终值（countdown 0.5667）并列引用。`QWEN7B_COUNTDOWN_SETUP.md` §七
的中期分析（step 41）仍然有效，但"跑满后重算 ρ / cum_disp 定向性 / 臂间排序"这三项
**永久无法从本次运行得到**，需重跑。7B 的臂间对比（baseaxis vs vanilla）因此仍不成立。
