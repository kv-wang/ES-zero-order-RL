# Countdown 线正式结果：Qwen2.5-1.5B-Instruct (2026-08-16)

**状态：PARTIAL**（ES momentum 未运行；单 seed；见下方注意事项）

## 问题

在有 ES 信号的 countdown regime（ρ=0.51，来自07-30的3B-Instruct测量）上，
用1.5B-Instruct模型，generation数对齐的情况下，GRPO vs ES vanilla vs ES baseaxis 的准确率如何？

## 配置

| 参数 | 值 |
|------|----|
| 模型 | Qwen/Qwen2.5-1.5B-Instruct |
| 任务 | countdown（训练集 rows[300:]，评测 rows[:300]） |
| ES N | 30 |
| ES B | 100 |
| ES steps | 100 |
| ES generations/arm | 300,000 |
| GRPO rollouts/step | 128（TRAIN_BS=16 × GROUP=8） |
| GRPO steps | 2343（= floor(300000/128)） |
| GRPO total_gen | 299,904（差96，0.03%） |
| σ | 1e-3 |
| α | 5e-4 |
| max_tokens | 2048 |
| eval cap | 300 |
| seeds | 单seed=0（低于3-seed预注册规则） |
| 启动时间 | 2026-08-15 10:37 |
| 完成时间 | 2026-08-16 11:32 |

## 结果

### 主表

| 方法 | countdown(ID) | extract_rate(cd) | math500(OOD) | OOD-avg | KL×1e³ | s/step |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| base（未训练） | 0.010 | 0.753 | 0.513 | 0.463 | — | — |
| GRPO gen-matched | **0.400** | 0.503 | 0.430 | 0.429 | -0.003 | — |
| ES vanilla | 0.240 | 0.490 | 0.500 | 0.450 | 8.2 | 217.3 |
| ES baseaxis | 0.183 | 0.640 | 0.467 | 0.412 | 35.7 | 216.7 |

OOD-avg = math500 / gsm8k / svamp / minerva_math / olympiadbench 的均值。

### 各方法完整OOD明细

| 方法 | math500 | gsm8k | svamp | minerva | olympiad | amc23 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| base | 0.513 | 0.750 | 0.787 | 0.096 | 0.120 | 0.275 |
| GRPO | 0.430 | 0.667 | 0.817 | 0.099 | 0.133 | 0.175 |
| ES vanilla | 0.500 | 0.737 | 0.763 | 0.085 | 0.167 | 0.325 |
| ES baseaxis | 0.467 | 0.667 | 0.713 | 0.107 | 0.107 | 0.175 |

## 注意事项与局限性

1. **单seed**：seed=0，低于本仓库3-seed预注册规则。单臂间差异（GRPO 0.400 vs vanilla 0.240）不足以支撑「哪个变体更好」的结论，仅能说明方法在该regime能产生非零信号。

2. **ES momentum 未运行**：tmux日志显示 `SKIP stage 4 (ES momentum)`，原因不明（可能是STAGES环境变量被覆盖）。三臂比较缺失最后一臂。

3. **countdown extract_rate 低**：vanilla=0.490、GRPO=0.503——超过一半的回答未按 `<answer>` 格式输出。准确率混合了「不会做」和「没按格式答」两种失败，extract_rate上升时需把格式学习与推理能力分开陈述。

4. **模型与07-30不同**：ρ=0.51是在3B-Instruct上测量的；本次用1.5B-Instruct，未测split-half ρ，不确认1.5B在此regime信号是否同样可分辨。

5. **vanilla的steps_scale_flagged异常**：100步中95步被标记为scale_flagged，含义需核查（可能是fitness variance过低触发了某个内部标志）。

6. **baseaxis KL异常高**：KL×1e³=35.7，远高于vanilla的8.2。baseaxis本应约束移动，但累计位移 final_cum_disp_frac=-1.69（即向theta0方向净移动了1.69倍轴范数），方向与预期相反，原因未查明。

7. **OOD全部下降**：GRPO和ES训练后math数学集OOD准确率均低于base（与八月数学线结论一致）。countdown是ID任务，math集下降属预期，不构成额外损伤证据；但与七月3B-Instruct结果的方向性差异（七月ES胜出）尚未解释。

8. **rc=1为脚本退出码bug**：已通过查看log确认vanilla和baseaxis均正常完成，rc=1来自 `[[...]] && ...` 短路求值覆盖 `$?`，不代表训练失败。

## 未解决问题

- ES momentum 为何被跳过？是否需要补跑？
- 1.5B-Instruct 的 countdown split-half ρ 是多少？若 ρ≈0 则当前结果无意义。
- vanilla的steps_scale_flagged=95/100是什么含义？
- baseaxis KL反向偏大的机制？
- 需要至少2个额外seed才能支持任何跨方法比较。
- GRPO wall_clock=45,777s（12.7h）vs ES vanilla=21,735s（6.0h）：wall-clock下GRPO慢2x，但generation数对齐。

## 产物路径

- GRPO summary: `axis_probe/results/grpo_qwen25_15b_instruct/grpo_countdown_gen2343_qwen25_15b_instruct_summary.json`
- ES vanilla: `axis_probe/results/countdown_qwen25_15b_instruct/vanilla_N30_s0_summary.json`
- ES baseaxis: `axis_probe/results/countdown_qwen25_15b_instruct/baseaxis_N30_s0_summary.json`
- 驱动日志: `axis_probe/logs/countdown_qwen25_15b_instruct/driver.log`
