# Contamination ledger (Track B eval hygiene)
Track-B paraphrases are an answer-preserving surface-perturbation family. Datasets that share this
perturbation family are reported in a SEPARATE FLAGGED column, never mixed into the primary OOD read.

| dataset | role | shares paraphrase family? | column |
|---|---|---|---|
| gsm8k (clean test) | ID | n/a | primary |
| svamp | OOD drift-axis endpoint | no | primary |
| asdiv | OOD drift-axis endpoint | no | primary |
| gsm_symbolic | perturbation-family OOD | YES | flagged (separate) |
| gsm_plus | perturbation-family OOD | YES | flagged (separate) |

## 测量有效性 (measurement validity) — 2026-08-08 追加

污染不是唯一会让 eval 读数失真的东西。以下是已确认的**测量层面**失效，
处置方式与污染相同：受影响的数字不得进入 primary 读数。

| 问题 | 受影响数据集 | 影响 | 状态 |
|---|---|---|---|
| `BOXED_RE` 无法匹配嵌套花括号 | math500 (20.7%)、minerva_math (25.0%)、olympiadbench (34.0%) | 这些题 gold 含 `{`，任何抽取路径都产不出含 `{` 的串 → **恒判 0**，与模型输出无关 | **未修复** — 见 [EXTRACTOR_BUG_REPORT](../../EXTRACTOR_BUG_REPORT.md) |
| 同一正则用于训练 gold 抽取 (`data_math.py:15`) | MATH L3-5 训练池 | 抽不到即 `continue`，**静默丢弃 23.6%**（1317/5586），保留池偏向数值型答案 | **未修复** |
| `used_boxed` 未持久化 (`eval_core.py:88`) | 全部 | artifacts 中无法区分 tier-1 (`\boxed`) 与 tier-3（末尾数字兜底）抽取 | **未修复** |

**读数规则：**
- **配对检验（McNemar 等）不受影响** —— 不可评分题对所有 arm 恒为 concordant，本就不计入。
  已实测验证剔除前后 z 值逐位相同。
- **绝对准确率、OOD-avg、以及任何 arm-vs-base 的边际比较受影响**，修复重跑前不得对外引用。
- svamp / gsm8k / countdown **不受影响**（答案为纯数值，或不经过该抽取器）。
