# 3B countdown 单卡 ES：vanilla vs baseaxis，两臂各跑满 100 次更新

**日期**: 2026-08-30
**问题**: 在唯一已知有成员排序信号的 regime（countdown）上，base-axis 探针是否优于纯高斯 vanilla ES？
**结论**: **不优于，而且显著更差。** countdown 末态 vanilla **0.4700** vs baseaxis **0.3933**（同一份 n=300 评测切片，McNemar 配对 30 vs 7，z=3.62，**p = 1.9e-4**）。两臂都远高于 base 0.0800。六个数学 OOD 集上 baseaxis 的退化也一致大于 vanilla。
**附带的仪表发现（本轮最有价值的一条）**: baseaxis 表面上 ρ 更高（0.3003 vs 0.2516），但这是**探针成员制造的测量假象**——剔除 4 个探针后只剩 0.2078，反而**低于** vanilla（逐步配对 Δ=+0.0438，se 0.0200，t=2.19，n=100）。

> 本报告取代 `COUNTDOWN_3B_SINGLE_GPU.md`。那一轮（08-27 启动）因评测门控 off-by-one 被 kill，其 vanilla 臂的 "step 90" 实为第 91 次更新，且无末态测量。本轮是修复后从 stage 0 重跑的完整两臂。

---

## 设置

| 项 | 值 |
|----|-----|
| 模型 | `Qwen/Qwen2.5-3B-Instruct` |
| 硬件 | 单张 H200 NVL（143,771 MiB），GPU 0，**TP=1** |
| 驱动 | `es_bench/axis_probe/run_countdown_3b.sh`（tmux session `es_3b_countdown`） |
| ES | N=30，B=100，steps=100，σ=1e-3（恒定，`sigma_adapt` 未启用），α=5e-4，mini_batch=64 |
| 生成次数 | 100 × 30 × 100 = **300,000** / 臂 |
| 评测 | 每 **10 次更新** + 末态，cap=300，max_tokens=2048（train == eval，贪心） |
| vLLM | `gpu_memory_utilization=0.70` |
| 臂 | stage 0 base → stage 2 vanilla → stage 3 baseaxis（串行同卡） |
| seed | **仅 s0（单 seed）** |

时间线：base 08-28 22:00:53 起 148 s；vanilla 22:03:21 → 08-29 13:04:28（driver 侧 54,067 s，trainer 内 53,933 s = 14.98 h）；baseaxis 13:04:28 → 08-30 04:12（54,468 s = 15.13 h）；stage 4 于 08-30 04:14:57 `ALL DONE`。

---

## countdown 曲线（n=300，ID 任务）

**本轮的横轴是真实的「已提交更新次数」**，不是循环下标。10…100 十个点全部落地，`eval_final` 也存在。

| 更新次数 | base | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | **100** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **vanilla** acc | 0.0800 | 0.1400 | 0.2133 | 0.2367 | 0.3233 | 0.3367 | 0.4200 | 0.4367 | 0.4133 | 0.4667 | **0.4700** |
| vanilla extract | 0.8900 | 0.7733 | 0.7267 | 0.6067 | 0.5267 | 0.5633 | 0.6033 | 0.5467 | 0.5100 | 0.5633 | 0.5633 |
| **baseaxis** acc | 0.0800 | 0.1500 | 0.1967 | 0.3167 | 0.3833 | 0.3900 | 0.3967 | 0.4167 | 0.4133 | 0.4100 | **0.3933** |
| baseaxis extract | 0.8900 | 0.7933 | 0.6967 | 0.5800 | 0.5333 | 0.5133 | 0.4933 | 0.5567 | 0.5367 | 0.5567 | 0.5500 |

形状差异明确：baseaxis 在 30–50 次更新区间领先（0.3167/0.3833/0.3900 对 0.2367/0.3233/0.3367），但在 **50 次更新之后基本停住**（0.39 → 0.3933，净 +0.003），而 vanilla 继续爬（0.3367 → 0.4700，+0.133）。末段 60–100 baseaxis 甚至轻微回落。

### `eval_final` 与曲线末点逐位一致

两臂的 `eval_final`（循环结束后另起一次完整评测）与第 100 次更新的曲线点**完全相同**：vanilla 0.4700 / extract 0.5633，baseaxis 0.3933 / extract 0.5500。同进程、同 θ、贪心解码下 vLLM 是可复现的；驱动自动生成的 `REPORT.md` 里那句「expect +/-1-2 points run to run」在**同进程复测**这个场景下未被观察到（跨进程未测）。

---

## 末态三方对比（cap=300，同一份评测切片）

| 集合 | n | base | vanilla | baseaxis | Δ(van−base) | Δ(base_ax−base) |
|---|---|---|---|---|---|---|
| **countdown（ID）** | 300 | 0.0800 | **0.4700** | 0.3933 | **+0.3900** | +0.3133 |
| math500 | 300 | 0.6333 | 0.6167 | 0.5533 | −0.0166 | −0.0800 |
| math500 L3–5（主指标） | 165 | 0.5576 | 0.5346 | 0.4793 | −0.0230 | −0.0783 |
| gsm8k | 300 | 0.8667 | 0.8700 | 0.8333 | +0.0033 | −0.0334 |
| svamp | 300 | 0.9133 | 0.9000 | 0.8933 | −0.0133 | −0.0200 |
| minerva_math | 272（全量） | 0.1912 | 0.1728 | 0.1434 | −0.0184 | −0.0478 |
| olympiadbench | 300 | 0.2533 | 0.2233 | 0.1933 | −0.0300 | −0.0600 |
| amc23 | 40（全量） | 0.4000 | 0.4000 | 0.2250 | 0.0000 | −0.1750 |

**读法**：vanilla 复现了上一轮「无灾难性遗忘」的结论——七个 Δ 全在 ±0.03 内，逐点都不显著。baseaxis 则在**全部六个数学集上都比 vanilla 差**（6/6 同向，符号检验 p = 0.031），单点仍多数不显著（amc23 n=40，±1σ ≈ 0.078，−0.175 约 2.2σ 是最强的一个），但方向一致性本身是信号。这与 countdown 上 baseaxis 更差是同向的：**baseaxis 不是「牺牲 OOD 换 ID」，而是两头都略差。**

---

## 配对显著性（McNemar，per-question jsonl，三臂同一 300 题且已核对逐题对齐）

| 对比 | 前者独对 | 后者独对 | 不一致数 | z（连续性校正） | p（精确二项） |
|---|---|---|---|---|---|
| base vs vanilla | 8 | 125 | 133 | 10.06 | 3.8e-28 |
| base vs baseaxis | 10 | 104 | 114 | 8.71 | 7.2e-21 |
| **vanilla vs baseaxis** | **30** | **7** | **37** | **3.62** | **1.9e-4** |

vanilla 与 baseaxis 差 23/300 = +0.0767。非配对合并标准误是 ±0.0403（约 1.9σ，不显著）；**配对后 p = 1.9e-4**，因为两臂在 300 题上高度重合，配对检验把共同方差消掉了。这是本轮唯一达到显著的臂间结论，也说明**臂间比较必须走配对路径**，非配对标准误在这个设计下过于保守。

---

## 抽取率（未解决的老混淆，本轮有新证据）

| 臂 | accuracy | extract_rate | 抽取成功内的条件正确率 |
|---|---|---|---|
| base | 0.0800 | 0.8900 | 0.0899 |
| vanilla | 0.4700 | 0.5633 | **0.8343** |
| baseaxis | 0.3933 | 0.5500 | **0.7152** |

extract_rate 从 0.890 崩到 0.55–0.56 的现象复现了，两个 ES 臂的崩塌幅度几乎一样。非抽取样本一律记错，所以 0.4700 / 0.3933 都只能当**下界**。

**新证据：这个混淆不能解释臂间差距。** 两臂 extract_rate 几乎相同（0.5633 vs 0.5500，差 0.0133），但抽取成功样本内的条件正确率差 **0.8343 vs 0.7152**，即 vanilla 的优势存在于「格式合规的那部分回答」内部，不是靠多抽出几条。仍未区分的是绝对水平的两种解释：(a) 真学会解题同时格式漂移，(b) 抽取器对新格式误判。需要人工抽查 per-question 文本，本轮未做。

---

## 仪表发现：base-axis 的 ρ 优势是探针成员制造的假象

`rho_curve.py` 默认对全部 30 个成员算 split-half Spearman。baseaxis 的前 4 个成员是探针（`use_probe` 为真时索引 0–3），它们沿 θ₀−θ_t 摆放，是种群里的系统性离群点。离线剔除这 4 个后重算：

| 臂 | ρ（全部 30 成员） | ρ（仅高斯成员） | 前 50 步 | 后 50 步 |
|---|---|---|---|---|
| vanilla | 0.2516 | 0.2516（无探针，定义相同） | 0.2539 | 0.2494 |
| baseaxis | **0.3003** | **0.2078** | 0.2962 | 0.3043 |

逐步配对（两臂每步用**同一批** B=100 题，`step_batches` 由 `C.DATA_SEED` 决定、与 `pop_seed` 无关）：vanilla − baseaxis 的高斯-only ρ 差 = **+0.0438**，sd 0.2003，se 0.0200，**t = 2.19，n = 100**。

**含义**：探针把 ρ 抬高了 0.09，但真正驱动梯度高斯分量的那 26 个成员之间的可分辨度反而更低。这与准确率结论方向一致（baseaxis 更差），并且直接证实了「ρ 读数可以被与梯度质量无关的种群结构污染」——**probe 臂与 vanilla 臂的 ρ 不同口径，历史上所有 probe-vs-vanilla 的 ρ 对比都需要按这个方式重算才可比**（含 7B countdown 那轮的 0.2698 vs 0.2287）。

**未按此重算的旧数字不要再引用。**

### ρ 随训练衰减：在 3B 上未复现

7B vanilla countdown 那轮 ρ 从前 50 步的约 0.286 掉到后 50 步的约 0.172（−40%）。3B 本轮**两臂都是平的**（vanilla 0.2539 → 0.2494，baseaxis 0.2962 → 0.3043）。所以「σ 恒定导致 ρ 随策略变好而衰减」这个解释**在 3B 上没有支持**，此前基于 7B 提出的「σ 反向自适应」动机相应减弱，需要先弄清两个模型规模上行为不同的原因。

---

## 机制诊断

| 量 | vanilla | baseaxis |
|---|---|---|
| s/step（trainer 内均值） | 539.3 s | 544.7 s |
| wall clock | 53,933 s = 14.98 h | 54,468 s = 15.13 h |
| peak_mem_alloc | 117.7 GB | **133.7 GB**（容量的 **93%**） |
| zero_update_rate | 0.0 | 0.0 |
| steps_probe_active | 0 | 95 / 95（全部计时步；warmup 5 步不计） |
| pair_tie_rate | — | 0.0474（B=100 下打平几乎消失，对比 B=8 时的 ~67%） |
| final_axis_norm ‖θ₀−θ_t‖ | 0（无轴） | 113.9 |
| final_cum_disp | 0（定义为 0） | **−87.43** |
| final_cum_disp_frac | 0 | **−1.8212** |
| steps_scale_flagged | 95（对 vanilla 无意义，见下） | 4 |

- **探针一致地投票「远离 θ₀」**：`signed_disp` 的符号约定是「+ 表示朝 θ₀ 靠近」，累计 −87.43 / 尺度无关的 −1.82 说明 100 步里探针几乎每步都把更新推离 base。这与 2026-07-23 那轮读到的「ES 主动往 base 走」方向相反，与 `FINAL_REPORT.md` 中抽取器修复后两个 seed 转负的结果一致。**在有信号的 regime 里，「撤销累积动量」这个方向被明确否决**——而 base-axis 每步花 4/30 个成员去问的正是这个问题，这解释了它为什么是净损失。
- `steps_scale_flagged=95` 对 vanilla 无意义：vanilla 无探针，`probe_pert_norm=0` 使 `scale_ratio=Infinity` 恒定触发。baseaxis 只有 4 步被 flag（早期 ‖轴‖ 尚小时）。
- **显存**：baseaxis 峰值 133.7 GB，只剩约 10 GB 余量（多出的约 16 GB 是 `axis_worker._theta0` 的 fp32 副本）。**在这张卡上不要再提高 util 或 N**。

---

## 本轮同时验证的代码修复

上一轮暴露的四处驱动 bug + 一处 trainer bug 全部生效：

1. `es_train_axis.py:382` 评测门控 `step > 0 and step % interval` → `(step + 1) % interval`：曲线现在落在第 10/20/…/100 次更新，per-question 前缀为 `_eval_upd###`，记录同时带 `updates` 和 `step`。
2. stage 2/3 补 `--eval_final`：两臂 summary 均有 `eval_final` 键，`report_countdown_paperB.py` 的 `acc()`/`extract_rate()` 不再全列 `--`。
3. rho 调用改 `--jsonl`/`--splits`：两臂 rho 阶段 **rc=0**，产出 `*_rho.json`（各 100 步）。
4. stage 4 reporter 按 `run_countdown_7b.sh` 那版重写：`REPORT.md` 正常生成，与 7B 表格列一致。
5. `S_PER_GEN` 0.15 → 0.18：预测 15.0 h/臂 vs 实测 14.98/15.13 h，**误差 < 1%**。

`python3 -m py_compile axis_probe/src/es_train_axis.py` 与 `bash -n axis_probe/run_countdown_3b.sh` 均通过（PY_OK / SH_OK）。

**遗留的小瑕疵**：驱动收尾的计数行打印 `base: 0 seed(s)`，因为它 glob 的是 `base_*_summary.json` 而实际文件名是 `base_summary.json`（第 258 行）。纯显示问题，不影响任何结果。

---

## 未解决 / 混淆

1. **单 seed（s0）**。所有臂间显著性只计入了抽样噪声，未计运行间方差。08-13 的功效分析测得 seed 间散布（~0.013 ID）与方法间差异同量级；本轮臂间差 0.0767 大于该量级，但**要作为可发表结论仍需 ≥2 seed**。
2. **N=26 vanilla 对照仍然缺失，且现在是承重混淆**。baseaxis 把 4/30 个成员挪给探针，所以「baseaxis 更差」无法区分「探针机制有害」与「高斯探索者少了 13%」。baseaxis 输了之后这条比它赢的时候更要紧——预注册里列过这个对照，一直没跑。
3. **抽取率崩塌的绝对水平未定论**（见上）。臂间差距已排除该解释，但 0.4700 / 0.3933 本身仍是下界。
4. **ρ 衰减在 3B 与 7B 上行为不一致**，原因未知。
5. **无 checkpoint**：ES 臂不落权重（`es_worker.py:80-83` 有 `es_save_base` 但 trainer 未接线），无法事后复评或续训。
6. **与 7B 那轮不可控比**：模型规模不同；且 7B 的 ρ 数字尚未按本报告的高斯-only 口径重算。

---

## 产物

- 结果目录：`experiments/2026-08-08_momentum_axis/es_bench/axis_probe/results/countdown_3b_N30_B100/`
  - `REPORT.md`（驱动自动生成）、`base_summary.json`
  - `vanilla_N30_B100_s0_{summary.json, evalcurve.jsonl, jsonl, rho.json}`
  - `baseaxis_N30_B100_s0_{summary.json, evalcurve.jsonl, jsonl, rho.json}`
  - 各集合 per-question jsonl（`*_finaleval__countdown_perq.jsonl` 等，McNemar 的输入）
- 日志：`.../logs/countdown_3b_N30_B100/`（`driver.log`、两臂 `.log`、`stage_times.jsonl`、`gpu_mem.csv`）
- 驱动：`.../axis_probe/run_countdown_3b.sh`

## 下一步（按优先级）

1. **补 seed（≥2）**——臂间结论目前是单 seed 的 p=1.9e-4，缺运行间方差。约 30 GPU-h/seed（两臂）。
2. **跑 N=26 vanilla 对照**，把「探针机制」与「探索者变少」分开。约 15 GPU-h。
3. **把高斯-only ρ 做成 `rho_curve.py` 的一个开关**（`--exclude_probes`），并按此口径重算 7B countdown 两臂，使历史 ρ 数字可比。零 GPU。
4. **查 countdown per-question 文本**，判定抽取率崩塌属于哪种解释；不做的话所有 countdown 数字只能以下界形式引用。零 GPU。
5. 弄清 ρ 衰减为何在 7B 出现而 3B 不出现，再决定 σ 自适应方向。
