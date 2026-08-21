# ES Baseaxis 实验报告：Qwen2.5-3B-Instruct

**实验日期：** 2026-08-18  
**实验者：** Caitlyn Yin  
**状态：** ✅ DONE

---

## 研究问题

ES baseaxis变体（使用base axis方向引导搜索）相比vanilla ES，在相同训练预算下能否提升性能和训练稳定性？

---

## 实验设置

### 模型与配置
- **模型：** Qwen/Qwen2.5-3B-Instruct
- **ES变体：** Baseaxis (axis="base")
- **训练参数：**
  - Population size: 16
  - Training steps: 200
  - Seed: 0
  - Sigma: 0.001 (固定，无自适应)
  - GPU memory utilization: 0.70

### 评测数据集（6套数学数据集）
1. **GSM8K** (n=300): 小学数学应用题
2. **SVAMP** (n=300): 数学应用题变体
3. **MATH500** (n=300): 多难度数学竞赛题（Level 1-5）
4. **Minerva Math** (n=272): 高难度数学推理
5. **OlympiadBench** (n=300): 奥林匹克数学竞赛题
6. **AMC23** (n=40): 美国数学竞赛题

### 对比基准
- **ES Vanilla** (N=16, 200步, seed=0) - 同日完成

---

## 实验结果

### 准确率对比（ES Baseaxis vs ES Vanilla）

| 数据集 | ES Vanilla | ES Baseaxis | 差异 | 相对提升 |
|--------|-----------|-------------|------|---------|
| **GSM8K** | 85.00% | 86.33% | +1.33% | +1.56% |
| **SVAMP** | 87.67% | 90.00% | +2.33% | +2.66% |
| **MATH500** | 56.33% | 57.33% | +1.00% | +1.78% |
| **Minerva Math** | 15.44% | 17.28% | +1.84% | +11.92% |
| **OlympiadBench** | 21.00% | 21.67% | +0.67% | +3.19% |
| **AMC23** | 25.00% | 32.50% | +7.50% | **+30.00%** |
| **平均** | **48.41%** | **50.85%** | **+2.44%** | **+5.05%** |

### 训练诊断指标

| 指标 | ES Vanilla | ES Baseaxis | 差异 |
|------|-----------|-------------|------|
| **KL Proxy Drift** | 0.0278 | 0.0044 | **-84.2%** |
| 训练时长 | 193.7 分钟 | 194.3 分钟 | +0.3% |
| 秒/步 | 58.1 | 58.3 | +0.3% |
| Peak GPU Memory | - | 133.7 GB | - |

### Baseaxis特有指标
- **Final Axis Norm:** 43.24
- **Final Cumulative Displacement:** 56.40
- **Mean Signed Displacement Fraction:** 0.0052 (朝向theta0的位移)
- **Pair Tie Rate:** 48.97% (配对评估中的平局率)
- **Zero Update Rate:** 1.54%

---

## 关键发现

### ✅ 性能全面提升
1. **所有6个数据集均有提升**，平均提升5.05%（相对）
2. **高难度任务提升更显著**：
   - AMC23（最难）：+30.0%相对提升
   - Minerva Math（高难度推理）：+11.92%相对提升
3. 简单任务也有稳定提升（GSM8K +1.56%, SVAMP +2.66%）

### ✅ 训练稳定性大幅改善
1. **KL drift降低84.2%**（0.0044 vs 0.0278）
2. 模型偏移更小，更好地保持了原始能力
3. 没有出现vanilla中观察到的fitness大幅波动

### ✅ 计算效率几乎无损
1. 训练速度与vanilla几乎相同（58.3秒/步 vs 58.1秒/步）
2. Axis计算的额外开销可忽略不计（+0.3%）

---

## 注意事项与局限

### ⚠️ 实验范围限制
1. **单一随机种子（seed=0）**：结果可能受特定随机性影响
2. **单一模型**：仅在Qwen2.5-3B-Instruct上测试，泛化性未知
3. **固定超参数**：sigma=0.001, N=16未经调优，可能存在更优配置
4. **小规模population**：N=16是较小的种群规模

### ⚠️ 评测局限
1. **样本量限制**：大部分数据集cap到300题（除Minerva Math和AMC23）
2. **单次推理**：每题仅一次生成，无majority voting
3. **答案提取率**：部分数据集存在答案提取失败（如OlympiadBench 78.67%）

### ⚠️ 对比不完整
1. **缺少Base baseline**：无法确认baseaxis是否优于未训练的base模型
2. **缺少GRPO对比**：无法确认baseaxis相比GRPO的相对表现
3. 根据vanilla报告，vanilla全面低于Base和GRPO，baseaxis是否能逆转这一趋势？

---

## 未解决的问题

1. **Baseaxis vs Base/GRPO的对比？**
   - Vanilla相比Base/GRPO全面下降
   - Baseaxis虽然优于vanilla，但是否仍低于Base？
   - 需要找到或重新运行Base和GRPO的评测结果

2. **为什么baseaxis优于vanilla？**
   - 是因为base axis提供了更好的搜索方向？
   - 还是因为减少了有害的参数扰动？
   - Mean signed disp frac=0.0052表明整体朝向theta0移动很小

3. **性能提升是否在统计上显著？**
   - 单seed结果，需要多seed验证
   - 2.44%的平均提升是否超出了随机噪声？

4. **高难度任务提升更大的原因？**
   - AMC23提升30%，简单任务仅1-3%
   - 是否因为base axis更适合保持复杂推理能力？

5. **Pair tie rate 48.97%意味着什么？**
   - 近一半的配对评估是平局
   - 这是否说明扰动效果太小？
   - 还是问题本身区分度不够？

---

## 后续建议

### 立即行动
1. **找到Base和GRPO的评测结果**，完成三方对比
2. 如果没有，在相同评测集上重新运行Base和GRPO评测

### 后续实验
1. **多seed验证**：至少3-5个不同seed，确认结果稳定性
2. **超参数扫描**：测试不同的sigma、N、训练步数
3. **其他模型验证**：在不同规模和系列的模型上测试
4. **其他axis变体**：测试momentum axis等其他方向

### 分析深化
1. **训练轨迹分析**：对比vanilla和baseaxis的fitness演化曲线
2. **参数空间分析**：检查最终模型的参数分布差异
3. **案例研究**：分析AMC23上提升最大的具体题目

---

## 文件产出

- **训练日志：** `baseaxis_N16_s0.jsonl` (200行)
- **总结文件：** `baseaxis_N16_s0_summary.json`
- **详细评测：** `baseaxis_N16_s0_finaleval__*.jsonl` (6个数据集)
- **结果目录：** `experiments/2026-08-08_momentum_axis/es_bench/axis_probe/results/momentum_qwen25_3b_instruct/`

---

## 结论

**ES Baseaxis在相同训练预算下全面优于ES Vanilla**，在所有6个数学评测集上平均提升5.05%，且训练稳定性显著改善（KL drift降低84%），计算开销几乎无增加。高难度任务（AMC23 +30%, Minerva Math +11.92%）的提升尤为显著。

然而，由于单一seed和缺少Base/GRPO对比，尚无法确定baseaxis是否真正改善了ES相对于传统方法的性能，还是仅仅在"两个都不好"的vanilla和baseaxis之间选出了"不那么差"的一个。完整的结论需要与Base/GRPO的三方对比。

**实验状态：技术上完成，但科学结论不完整。**
