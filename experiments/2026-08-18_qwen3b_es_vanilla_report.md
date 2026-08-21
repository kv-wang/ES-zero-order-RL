# Qwen2.5-3B-Instruct ES Vanilla 评测报告

**日期：** 2026-08-18  
**实验范围：** Qwen2.5-3B-Instruct，ES vanilla N=16，200步，单seed=0  
**产出：** `experiments/2026-08-08_momentum_axis/es_bench/axis_probe/results/momentum_qwen25_3b_instruct/`

## 研究问题

在Qwen2.5-3B-Instruct模型上，ES vanilla训练（N=16, 200步）相对Base模型和GRPO训练的表现如何？

## 实验设置

- **模型：** Qwen/Qwen2.5-3B-Instruct
- **训练方法：** ES vanilla（无momentum，无axis probe）
- **超参数：**
  - Population size: N=16
  - Training steps: 200
  - Population seed: 0（单seed）
  - Sigma: 0.001（固定，无自适应）
  - Learning rate (alpha): 未明确记录
  - GPU memory utilization: 0.70
- **评测数据集：** GSM8K, SVAMP, Math500, Minerva Math, OlympiadBench, AMC23
- **对照组：** Base模型（未训练），GRPO训练（400步，64 rollouts/step）

## 主要结果

### 三方对比：Base vs GRPO vs ES Vanilla

| 数据集 | Base | GRPO (400步) | ES Vanilla (200步) | ES vs Base | ES vs GRPO |
|--------|------|-------------|-------------------|-----------|-----------|
| **GSM8K** | 86.67% | 85.33% | **85.00%** | -1.67% | -0.33% |
| **SVAMP** | 91.33% | 91.67% | **87.67%** | -3.66% | -4.00% |
| **Math500** | 61.67% | 62.33% | **56.33%** | -5.34% | -6.00% |
| **Math500 L3-5** | 53.46% | 54.38% | **48.39%** | -5.07% | -5.99% |
| **Minerva Math** | 18.75% | 16.91% | **15.44%** | -3.31% | -1.47% |
| **OlympiadBench** | 23.33% | 21.00% | **21.00%** | -2.33% | 0% |
| **AMC23** | 35.00% | 42.50% | **25.00%** | -10.00% | -17.50% |

### 训练诊断指标

- **Wall clock time:** 11,622秒（约3.23小时）
- **平均每步耗时:** 58.1秒
- **KL proxy drift:** +0.0278（相比GRPO的-6.98e-08≈0，显示模型分布有一定偏移）
- **Zero update rate:** 0.51%（1次零更新）
- **Steps scale flagged:** 195/200（97.5%的步数被标记）
- **Peak memory:** 117.67 GB allocated, 124.24 GB reserved

### 答案提取率对比

| 数据集 | Base提取率 | GRPO提取率 | ES提取率 |
|--------|-----------|-----------|---------|
| Math500 | 91.33% | 97.67% | **89.67%** ↓ |
| Minerva Math | 90.07% | 95.96% | **91.54%** → |
| OlympiadBench | 79.00% | 90.67% | **81.33%** → |
| AMC23 | 75.00% | 92.50% | **82.50%** ↑ |

## 关键发现

### 1. **ES Vanilla全面性能下降**
- 所有数据集的准确率均低于Base模型
- 最大退化：AMC23 (-10% vs Base, -17.5% vs GRPO)
- 不像GRPO那样显著改善格式规范（答案提取率）

### 2. **训练不稳定的迹象**
- 97.5%的训练步数被标记为scale flagged
- KL漂移+0.0278（GRPO接近0）
- 零更新率虽然很低（0.51%），但scale flagging比例异常高

### 3. **与GRPO的对比**
- GRPO训练了400步（2倍），但ES只用了200步
- GRPO每步使用64 rollouts，信息量比ES的N=16更大
- GRPO的KL漂移接近0，模型分布几乎未改变
- GRPO在格式规范上有显著提升（提取率↑），ES没有

## 注意事项与局限

### ⚠️ **单一实验种子**
- 本实验仅使用单个population seed (s0)
- 无法估计seed间方差
- 不足以支撑统计显著性结论

### ⚠️ **模型特殊性**
- Qwen2.5-3B-Instruct本身是相对成熟的instruct模型
- 可能已处于局部最优点，微调空间有限
- 结论可能不适用于base模型或其他预训练模型

### ⚠️ **超参数未优化**
- Population size N=16可能偏小（论文使用N=30或更大）
- 训练步数200步可能不足（论文使用500-2500步）
- Sigma固定在0.001，未做自适应调整

### ⚠️ **训练设置差异**
- ES与GRPO训练步数不同（200 vs 400）
- ES与GRPO每步生成数不同（N=16 vs 64 rollouts）
- 不是严格控制变量的对比

### ⚠️ **评测限制**
- 部分数据集样本量小（AMC23仅40题）
- 大部分数据集capped到300题，可能低估真实性能
- 单次评测，无重复验证

## 未解决的问题

1. **为什么ES表现如此差？**
   - 是population size太小？
   - 是训练步数不足？
   - 是sigma设置不当？
   - 还是ES本身不适合已优化的instruct模型？

2. **Steps scale flagged的含义？**
   - 97.5%被标记意味着什么？
   - 是否指示训练过程异常？

3. **如何改进？**
   - 增大N到30或更大？
   - 延长训练到500步？
   - 使用sigma自适应？
   - 切换到base模型而非instruct模型？

4. **与论文结果的差距？**
   - 论文中ES在数学任务上有效
   - 本实验ES完全失败
   - 差异来自哪里？

## 后续建议

1. **多seed验证**：至少运行3个不同的population seeds
2. **参数扫描**：测试不同的N（如30, 64）和steps（如500）
3. **切换模型**：在base模型上测试，避免instruct模型的局部最优陷阱
4. **增加batch size**：当前每个成员的奖励批量可能太小
5. **与baseaxis对比**：等待baseaxis实验完成后进行横向对比

## 产出文件

- Summary: `vanilla_N16_s0_summary.json`
- Training log: `vanilla_N16_s0.jsonl` (200行，每步一行)
- Evaluation results (per-question): `vanilla_N16_s0_finaleval__*.jsonl` (6个数据集)
