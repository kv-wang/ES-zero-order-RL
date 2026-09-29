# 3B countdown 单卡 ES：vanilla 臂跑满 100 步（baseaxis 臂进行中）

> **已被取代（2026-08-30）**：本轮 baseaxis 臂被 kill，且 vanilla 曲线的「step 10/20/…/90」实为第
> 11/21/…/91 次更新、无末态测量（见「未解决」第 3 条）。修复后从 stage 0 重跑的完整两臂结果见
> `COUNTDOWN_3B_VANILLA_VS_BASEAXIS.md`。本文件保留作为 bug 诊断记录，其数字不要再作为结论引用。

**日期**: 2026-08-28
**问题**: 7B GRPO 单卡不可行（见 `GRPO_7B_SINGLE_GPU_FAILURE.md`）之后，换成 Qwen2.5-3B-Instruct 单卡跑 countdown ES，**vanilla ES 能否在单卡上把 countdown 从 base 水平推起来**，以及 3B 单卡这条路的墙钟/内存是否可持续。
**结论（仅 vanilla 臂，单 seed）**: countdown 从 base **0.080 → 0.470**（step 90，n=300，Δ = +0.390，非配对合并标准误 ±0.033，约 12σ）。六个数学 OOD 集在 ±1σ 内基本不动。**但 countdown 的抽取率同时从 0.89 崩到 0.50–0.56，这是本次最大的未解决混淆**，0.470 只能当下界读。

---

## 设置

| 项 | 值 |
|----|-----|
| 模型 | `Qwen/Qwen2.5-3B-Instruct`（2 shards，5.75 GB） |
| 硬件 | 单张 H200 NVL，GPU 0，**TP=1**（无张量并行） |
| 驱动 | `es_bench/axis_probe/run_countdown_3b.sh`（tmux session `es_3b_countdown`, pts/7） |
| ES | N=30，B=100，steps=100，σ=1e-3，α=5e-4，mini_batch=64 |
| 生成次数 | 100 × 30 × 100 = **300,000** / 臂（与 GRPO 那侧 300,000 的预算对齐） |
| 评测 | 每 10 步，cap=300，max_tokens=2048（train == eval，贪心） |
| vLLM | `gpu_memory_utilization=0.70` |
| 臂 | stage 0 base → stage 2 vanilla → stage 3 baseaxis（串行，同一张卡） |
| seed | **仅 s0（单 seed，无运行间方差）** |

时间线：base 2026-08-27 23:15:39 起 136 s 完成；vanilla 23:17:55 → 2026-08-28 14:15:48，**elapsed 53,873 s = 14.96 h**；baseaxis 14:15:49 启动，本报告写作时在 step 10 附近。

---

## vanilla 臂结果

### countdown（ID 任务，n=300）

| step | base | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
|------|------|----|----|----|----|----|----|----|----|----|
| accuracy | 0.080 | 0.130 | 0.230 | 0.253 | 0.350 | 0.337 | 0.427 | 0.433 | 0.420 | **0.470** |
| extract_rate | 0.890 | 0.747 | 0.737 | 0.587 | 0.597 | 0.570 | 0.597 | 0.557 | **0.497** | 0.560 |

- Δ(step 90 − base) = **+0.390**。非配对合并标准误 = √(0.47·0.53/300 + 0.08·0.92/300) = **±0.033** → 约 **11.9σ**。
- 曲线单调性不完美但方向清楚：40→50 与 70→80 各回落 0.013 / 0.013，均远小于 ±0.029 的单点标准误。
- **配对检验未做**：两次评测用的是同一份 cap=300 子集，per-question jsonl 都在盘上（`base_finaleval__countdown_perq.jsonl` 与 `vanilla_*_eval_step0*__countdown_perq.jsonl`），McNemar 配对检验可离线补算且会比 ±0.033 更紧，本轮没跑。

### 六个数学集 OOD（step 90 vs base）

| 集合 | n | base | step 90 | Δ | 该点 ±1σ |
|------|---|------|---------|---|----------|
| math500 | 300 | 0.6333 | 0.6133 | −0.020 | ±0.028 |
| math500 L3–5（主指标） | 165 | 0.5576 | 0.5346 | −0.023 | ±0.039 |
| gsm8k | 300 | 0.8667 | 0.8700 | +0.003 | ±0.019 |
| svamp | 300 | 0.9133 | 0.8967 | −0.017 | ±0.017 |
| minerva_math | 272（全量） | 0.1912 | 0.1801 | −0.011 | ±0.023 |
| olympiadbench | 300 | 0.2533 | 0.2267 | −0.027 | ±0.025 |
| amc23 | 40（全量） | 0.4000 | 0.4500 | +0.050 | ±0.078 |

**读法**：**没有灾难性遗忘**。七个 Δ 全落在各自 ±1.1σ 内，逐点都不显著。方向上 4 降 2 升 1 平（符号检验不显著），所以「countdown 涨了 0.39 而数学掉一点」这句话只能当趋势说，不能当结论；要判定必须多 seed 或全量评测。

### 机制与开销

| 量 | 值 |
|----|-----|
| s/step（均值） | **538.2 s** → 0.179 s/gen |
| wall clock | 53,820 s = 14.95 h（驱动侧含启动 53,873 s） |
| peak_mem_alloc | **117.67 GB**；reserved 120.98 GB（占 143.77 GB 容量 **81.9% / 84.1%**） |
| zero_update_rate | 0.0（100 步无空更新） |
| σ | 恒定 1e-3（`sigma_adapt=false`，ratio min=max=1.0） |
| final_cum_disp | 0.0（vanilla 无 axis，按定义为 0） |

- **单卡内存有 ~23 GB 余量**，3B 单卡这条路可持续；对照 7B GRPO 单卡三档 bs 全在 126–139 GB OOM。
- **`S_PER_GEN=0.15` 默认值低估 20%**：实测 0.179 s/gen，驱动预测 13.5 h/臂，实际 14.96 h。脚本里该常数应改成 0.18。
- `steps_scale_flagged=95` 对 vanilla **无意义**：vanilla 没有 probe，`probe_pert_norm=0` 使 `scale_ratio=Infinity` 恒定触发该 flag，不是异常信号。

---

## 未解决 / 混淆

1. **抽取率崩塌（最重要）**：countdown extract_rate 0.89 → 0.497–0.56。0.470 的分子里，非抽取样本一律记错，所以 **0.470 是下界**；抽取成功样本内的条件正确率是 0.470/0.560 = **0.839**。两种解释没被区分开：(a) ES 真的学会解题，同时输出格式漂移到抽取器解析不了；(b) 部分「提升」来自抽取器对新格式的误判。**必须查 per-question jsonl 才能定论**，本轮没查。这条与仓库历史上那次 extractor 修复是同一类风险。
2. **单 seed（s0）**：无运行间方差估计，所有 Δ 的显著性只算了抽样噪声。
3. **最后一次评测在 step 90，不在 step 100**（两个独立原因叠加）：
   - **索引到不了 100**：`es_train_axis.py:263` 是 `for step in range(num_steps)`，num_steps=100 → step 只走 0–99；评测条件 `es_train_axis.py:373` 是 `step > 0 and step % eval_interval == 0`，最后一次命中在 step 90。评测放在 commit 之后，所以「step 90」这个点对应的是**已提交 91 次更新**的 θ，最后 9 次更新（step 91–99）无任何测量。
   - **末态评测的开关没打开**：`es_train_axis.py:111` 定义了 `--eval_final`，`es_train_axis.py:442` 会在循环结束后跑一次完整评测并写进 `summary["eval_final"]`（perq 前缀 `*_finaleval`）。但 `run_countdown_3b.sh` 的 stage 2/3 调用（第 191–199、220–228 行）只传了 `--eval_interval`/`--eval_cap`，**没传 `--eval_final`**——而驱动横幅却写着 `eval : every 10 steps + final`，横幅与实际命令不符。核对：`vanilla_N30_B100_s0_summary.json` 里确实没有 `eval_final` 键，`eval_history` 止于 step 90。
   - **后果**：vanilla 的 100 步终态权重未落盘（见第 5 条）也未评测，**不可事后补测**，只能重跑。要和别的臂/别的模型比，只能引用 step 90 这个点。
   - **对臂间比较无害**：baseaxis 用的是同一份调用，同样会停在 step 90，两臂末点仍然配对。
   - **标签本身也是错的**：step 0 就提交了一次真实更新（baseaxis 日志 step 0 有 `|D|=5.07e+00`，step 1 的 `axis=5.07` 即其范数），评测又在 commit 之后，所以循环下标 t 对应**已提交 t+1 次更新**。旧曲线的「step 10/20/…/90」实际是第 11/21/…/91 次更新。
   - **已修（2026-08-28，本轮跑完后生效）**：`es_train_axis.py:382` 的门控从 `step > 0 and step % interval == 0` 改为 `(step + 1) % interval == 0`，下标落在 9/19/…/99，即**第 10/20/…/100 次更新**，末点就是训练完的模型，`--eval_final` 在曲线意义上不再必需。记录同时写 `updates` 和 `step`，per-question 前缀改为 `_eval_upd###`。要求 `eval_interval` 整除 `num_steps`。**改动只对之后启动的 run 生效**：vanilla 已结束、baseaxis 启动时已把旧代码编译进内存，两臂仍是旧口径；新旧曲线不能按标签直接叠图。
   - **driver 侧待办**：`report_countdown_paperB.py:87-96` 只读 `summary["eval_final"]`，所以仍需二选一——stage 2/3 补 `--eval_final`（参照 `run_countdown_7b.sh:234`，那一版还传了 `--kl`，这是 7B 有末态评测、3B 没有的原因），或让 reporter 在缺该键时回退到 `eval_history[-1]`。`run_countdown_3b.sh` 是删改后的副本，把这个开关连同 rho、readout 的参数一起改错了，三处一起补。
4. **ρ 曲线缺失**：`rho_vanilla_s0` **rc=2、0 秒失败**——驱动把 jsonl 当位置参数并传 `--n_splits`，而 `rho_curve.py` 要求 `--jsonl` 和 `--splits`。fitness/correct_bits 都在 jsonl 里，**ρ 可离线补算**，但本轮没有 ρ 数字。同一处 bug 会让 baseaxis 的 ρ 阶段以同样方式失败。注意：驱动脚本此刻正被 bash 执行中，直接改文件有 bash 按字节偏移续读导致执行错乱的风险，**修脚本要等本次跑完**；ρ 可以另起进程离线算，不受影响。
5. **无 checkpoint**：ES 臂不落权重，跑完只有 jsonl / summary，无法事后复评或续训。
6. **臂间比较尚不成立**：baseaxis 仍在跑（14:15:49 启动，step 10 时 countdown 0.1633，ETA 约 15.2 h → 2026-08-29 05:30 前后）。step 10 这个唯一配对点 vanilla 0.130 vs baseaxis 0.163，Δ = +0.033 对 ±0.029 只有 1.15σ，**不构成任何结论**。
7. **与 7B 那轮的比较口径不齐（已修正）**：7B REPORT.md 里的 0.5667 取自 `eval_final`（100 次更新后的末态），而 3B 只有 step 90（91 次更新），此前直接对比是拿末态比中途。**同口径应取 7B 的 step 90 = 0.5300**（extract 0.7167）对 3B 的 0.4700（extract 0.560）：Δ = +0.060，合并标准误 ±0.041，约 **1.5σ，不显著**。参考：7B 自己从 step 90 的 0.5300 走到末态 0.5667，最后 9 次更新贡献 +0.0367（约 1.3σ）。即便同口径，模型规模仍不同，只能当量级参考。
8. **stage 4 readout 会失败（尚未触发）**：`run_countdown_3b.sh:241` 调 `report_countdown_paperB.py "$RES" --model_name "$MODEL"`，但该脚本 argparse 要求 `--results/--logs/--mem_csv/--stage_times/--es_generations/...`，既无位置参数也无 `--model_name` → 必然 rc=2；驱动又用 `> "$RES/REPORT.md" 2>&1`，会把 argparse 报错写进 REPORT.md。**即使参数改对，`report_countdown_paperB.py:87-96` 的 `acc()`/`extract_rate()` 只读 `eval_final`**，3B 两臂都缺这个键，表格会整列 `--`；要出 3B 的 REPORT.md，必须先补第 3 条的 `--eval_final`（需重跑），或改 reporter 回退到 `eval_history[-1]`。正确调用形式见 `run_countdown_7b.sh:261-268`。baseaxis 跑完（约 2026-08-29 05:30）即触发。

---

## 产物

- 结果目录：`experiments/2026-08-08_momentum_axis/es_bench/axis_probe/results/countdown_3b_N30_B100/`
  - `base_summary.json`、`vanilla_N30_B100_s0_summary.json`、`vanilla_N30_B100_s0.jsonl`（100 行）、`vanilla_N30_B100_s0_evalcurve.jsonl`（9 行 = step 10–90）、各集合 per-question jsonl
- 日志：`.../logs/countdown_3b_N30_B100/`（`driver.log`、`vanilla_N30_B100_s0.log`、`stage_times.jsonl`、`gpu_mem.csv`）
- 驱动：`.../axis_probe/run_countdown_3b.sh`，正式启动包装 `.../axis_probe/launch_countdown_3b_full.sh`

## 下一步

1. baseaxis 跑完后做 vanilla vs baseaxis 的同长度对比（step 90 配对点 + cum_disp 定向性）。
2. 离线补算两臂 ρ（用 `--jsonl`/`--splits` 正确参数）。跑完后一次性修 `run_countdown_3b.sh` 四处：rho 调用参数（第 204、233 行）、stage 2/3 补 `--eval_final`（第 196、225 行附近）、stage 4 的 reporter 调用（第 241 行，照 `run_countdown_7b.sh:261-268`）、`S_PER_GEN=0.18`。
3. 查 countdown per-question jsonl，判定抽取率崩塌属于哪种解释；这一步不做，0.470 就只能以下界形式引用。
4. 若要可发表的臂间结论，需补 seed（≥2）以获得运行间方差。
