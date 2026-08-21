# 显存测量 — 仪表补齐,以及一次失败的测量(2026-08-13)

> ## ⚠ 本轮测到的两个数字**均不可引用**
>
> ES 79.92 GB 与 GRPO 30.28 GB **不是同一类量**。差距几乎全部来自记账口径,
> 与方法本身的显存效率无关。根因已查明并验证(见 §根因)。
> **在完成 §正确测法 之前,本 try 不得引用任何显存比较。**

## 背景:本 try 此前没有显存数据

07-18 那轮的 `es_train.py` 记录完整——第 98 行调 `es_reset_peak_mem()`,结束时写**三个**口径:
`peak_mem_torch_gb` / `peak_mem_reserved_gb` / **`peak_mem_nvidia_smi_mib`**。
而本 try 的 `es_train_axis.py` 从未接线(`es_worker.py` 里两个方法都在,训练器不调用),
因此已完成的 10 个 ES arm-seed 显存数据全部缺失。

本轮补上了 torch 的两个口径,**但漏了 nvidia-smi 那个**——而事后证明只有它有效。

## 测到的数字(及其无效性)

| | ES(vanilla, N=16, 12 步) | GRPO(400 步) |
|---|---:|---:|
| `gpu_memory_utilization` | 0.50 | 0.50(rollout) |
| 峰值分配 | 79.92 GB | 30.28 GB |
| 峰值保留 | 82.33 GB | 41.83 GB |
| 数据来源 | `torch.cuda.max_memory_allocated()`,**vLLM 同进程内** | verl `actor/perf/max_memory_allocated_gb`,**actor 进程内** |

表面上 ES 是 GRPO 的 2.6 倍。**这个比较无效。**

## 根因:进程隔离(已验证)

GRPO 的日志显示三个进程:

```
(TaskRunner     pid=25333)   ← verl 主控,actor 指标在此测
(WorkerDict     pid=25656)   ← FSDP 训练 worker
(vLLMHttpServer pid=26369)   ← vLLM,rollout 引擎
```

**`torch.cuda.max_memory_allocated()` 按进程统计。** KV cache 分配在 pid 26369,
而指标在 TaskRunner/WorkerDict 侧读取——**另一进程的分配对 torch 计数器不可见**。
数字也自洽:util=0.5 应给 vLLM 约 71.9 GiB,若 30.28 GB 含 KV 则不可能小于它。

ES 恰好相反。训练器第 137 行显式设置:

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

**强制 vLLM 同进程**。这不是选择而是必需——ES 要用 `collective_rpc` 直接改写 vLLM worker
里的权重(`es_snapshot_base` / `es_set_member`),`worker_extension_cls` 注入的方法
只有同进程才能操作那份权重。所以 ES 的峰值**必然含满额 KV**。

### KV 主导的直接验证

跑两档 `gpu_mem_util`,看峰值是否随预算线性变化:

| util | 实测峰值 | 解析预期(`总显存×util + fp32 base 5.75 + 瞬时 ~2`) | 偏差 |
|---:|---:|---:|---:|
| 0.20 | 32.5 GiB | 35.9 GiB | −3.4 |
| 0.50 | 74.4 GiB | 78.0 GiB | −3.6 |

**两档偏差几乎相同(−3.4 / −3.6),峰值随 util 线性移动。**
确认 ES 的测量值由 KV 预算决定,**不反映方法需求**。

## 方法固有需求(解析估算,非实测)

抛开 KV:

| 组成 | ES | GRPO |
|---|---:|---:|
| fp16 live 权重 | 2.9 GB | 2.9 GB |
| fp32 base / master | 5.75 GB | 5.75 GB |
| 梯度 | — | ~5.75 GB |
| Adam 两个动量 | — | 11.5 GB |
| 激活值(已开 gradient checkpointing) | — | 有 |
| 瞬时缓冲 | ~2 GB(逐张量,最大 embedding 0.93 GB) | — |
| **合计** | **≈ 10.6 GB** | **≈ 26–30 GB** |

ES 约为 GRPO 的三分之一,方向上支持论文"只需前向所以省显存"的主张,
也与 GRPO 实测的 30.28 GB(纯 actor 侧)吻合。**但这是推算,不是实测,同样不可作为结论引用。**

## 正确测法:只有进程外观测

框架内部的记账口径**无法统一**——一个同进程、一个跨进程,而这是两种方法架构上的必然差异,
不是配置疏忽。因此唯一有效的做法是**在进程之外观测整卡**:

- 训练期间起一个采样线程轮询 `nvidia-smi --query-gpu=memory.used`,取最大值;
- 或对进程树的显存求和。

这正是 07-18 记录 `peak_mem_nvidia_smi_mib`(79619 MiB)的原因——绕开框架记账差异。
**我补仪表时只接了 torch 的两个口径,漏了这一个,是本轮测量失败的直接原因。**

要得到可比数字,需:(a)两边 KV 预算设成相同;(b)都用 nvidia-smi 峰值;
(c)各跑一次短探测(显存峰值在前几步即达到,不随步数增长,故 12 步足够)。

## 未解决

- **现有的 79.92 GB 与 30.28 GB 不得被引用**,包括不得用于任何"ES 更省/更费显存"的表述。
- nvidia-smi 峰值采样**尚未实现**。
- 已完成的 10 个 arm-seed 无显存数据;补测需重跑或以短探测替代。
- 单配置、无 seed 重复(显存对 seed 不敏感,此项影响小)。

## 产出

`axis_probe/results/memprobe/van_u05_summary.json`、`van_u020_summary.json`;
`es_train_axis.py` 新增 `--gpu_mem_util` 与 `peak_mem_alloc_gb` / `peak_mem_reserved_gb`
(**注:这两个字段对跨方法比较无效,仅可用于同方法内不同 util 的相对比较**)。
