# 实现 Log-likelihood Reward 需要添加的逻辑

## 目标

用生成文本的 log-likelihood 作为 reward，代替当前的 0/1 二值 reward：
```
reward = Σ log P(token_t | prefix_{<t})
```

这是**真正连续的** reward（范围 −∞ 到 0），能充分利用模型 logits 的连续信息。

## 核心挑战

### 1. 需要 Ground-Truth Trace

**Log-likelihood reward 需要知道"正确答案的 token sequence"**，才能计算它在模型下的概率。

当前 Countdown 数据集只有：
```python
{
  "numbers": [3, 5, 25],
  "target": 200,
  "question": "Use numbers 3, 5, 25 to get 200"
}
```

**没有标注的推理过程**，例如：
```
GT trace: "(5+3)*25"  # 我们不知道这个
```

即使有 GT，也可能有多个等价的解（`(5+3)*25` 和 `(3+5)*25` 和 `25*(5+3)` 都正确），需要决定用哪个。

### 2. vLLM 调用需要改动

当前代码：
```python
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42, stop=C.STOP)
outputs = llm.generate(prompts, sp)
texts = [o.outputs[0].text for o in outputs]
```

需要改成：
```python
sp = SamplingParams(
    temperature=0.0, max_tokens=max_tokens, seed=42, stop=C.STOP,
    logprobs=1  # 返回每个 token 的 top-1 logprob
)
outputs = llm.generate(prompts, sp)
```

vLLM 的 `RequestOutput` 对象会包含 `.outputs[0].logprobs`，是一个 `list[dict]`，每个 dict 是 `{token_id: Logprob对象}`。

### 3. Teacher-Forcing 模式

Log-likelihood reward 通常需要 **teacher-forcing**：给定 GT trace，在模型下计算其概率，而不是让模型 greedy decode。

vLLM 的 `logprobs` 参数**只返回实际生成的 tokens 的 logprob**，如果模型 greedy decode 出错了的路径，我们拿不到 GT trace 的 logprob。

要拿 GT trace 的 logprob，需要：
- **方案 A**：用 `prompt_logprobs` 把 GT trace 当作 prompt 的一部分，拿它的 prompt logprobs（但这不是 generation）
- **方案 B**：单独跑一次 forward pass，用 HF transformers 的 `model.forward(input_ids, labels=gt_ids)` 拿 loss
- **方案 C**：修改 vLLM，支持 teacher-forcing generation（返回给定 token sequence 的 logprobs）

**vLLM 不直接支持 teacher-forcing generation**，所以需要额外的工具链。

---

## 实现方案（按难度递增）

### 方案 A：混合 reward（Outcome + Log-likelihood）

**思路**：保持当前的 greedy generation，但用生成文本的 log-likelihood 作为 tiebreaker。

```python
# 生成时开启 logprobs
sp = SamplingParams(..., logprobs=1)
outputs = llm.generate(prompts, sp)

for o in outputs:
    text = o.outputs[0].text
    logprobs = o.outputs[0].logprobs  # list[dict[int, Logprob]]
    
    # 提取每个 token 的 logprob（greedy 下只有 1 个）
    token_logprobs = [list(lp.values())[0].logprob for lp in logprobs]
    total_logprob = sum(token_logprobs)  # 连续值，范围 (-∞, 0]
    
    # 混合 reward：correctness 为主，logprob 为辅
    correctness = answer_reward_function(text, numbers, target)  # {0, 1}
    reward = correctness + 0.01 * total_logprob  # 例如：1.0 + 0.01*(-50) = 0.5
```

**优点**：
- 不需要 GT trace
- 代码改动最小（只加一行 `logprobs=1` + 提取逻辑）
- 当答案全错时，logprob 高的（fluent but wrong）会比 logprob 低的（gibberish）得分高

**缺点**：
- 仍然是"答对得 1，答错得 <1"，logprob 只是连续的惩罚项
- 不是纯 log-likelihood reward（没有 GT）

**代码改动清单**：
1. `es_train_axis.py:250` — `SamplingParams` 加 `logprobs=1`
2. `es_train_axis.py:~330`（generation 循环）— 提取 `output.outputs[0].logprobs`，计算 `sum(token_logprobs)`
3. `data_countdown.py` — 新增 `reward_logprob(text, row, total_logprob)` 返回混合 reward
4. `es_train_axis.py` — 新增 `--logprob_weight` flag，控制 logprob 的权重

---

### 方案 B：用 HF Transformers 计算 GT Trace 的 Log-likelihood（需要 GT）

**思路**：假设我们有 GT trace，用 HuggingFace transformers 单独跑 forward pass 计算它的 NLL（negative log-likelihood）。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型（与 vLLM 共享权重，或单独加载）
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def compute_gt_loglikelihood(prompt, gt_trace):
    """计算 GT trace 在模型下的 log-likelihood（teacher-forcing）"""
    full_text = prompt + gt_trace
    input_ids = tokenizer.encode(full_text, return_tensors="pt")
    prompt_len = len(tokenizer.encode(prompt))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 只计算 GT trace 部分的 logprob（不包括 prompt）
    log_probs = torch.log_softmax(logits[0, prompt_len-1:-1, :], dim=-1)
    target_ids = input_ids[0, prompt_len:]
    token_logprobs = log_probs[range(len(target_ids)), target_ids]
    
    return token_logprobs.sum().item()  # 连续值 ∈ (-∞, 0]
```

**优点**：
- 真正的 log-likelihood reward（GT 在模型下的概率）
- 不依赖生成质量（即使模型 greedy decode 错了，GT 的 logprob 仍然是客观的）

**缺点**：
- **需要 GT trace**（Countdown 数据集没有）
- 每个 member 生成后，要额外跑一次 forward pass（2× 计算量）
- 需要在 GPU 上同时持有 vLLM 的模型（fp16 weights）和 HF 的模型（或共享权重）

**代码改动清单**：
1. **数据准备**：人工标注或用 GT solver 生成一些 Countdown 问题的推理 trace（例如 1000 条）
2. `data_countdown.py` — 加载 GT traces，新增 `gt_trace(row)` 函数
3. `es_train_axis.py` — 加载 HF 模型，在生成后调用 `compute_gt_loglikelihood(prompt, gt_trace(row))`
4. 新增 `--use_gt_loglikelihood` flag

---

### 方案 C：修改 vLLM，支持 Teacher-Forcing（最难）

**思路**：在 vLLM 里加一个 API，给定 `(prompt, target_tokens)`，返回 target_tokens 的 logprobs（不生成）。

```python
# 伪代码，vLLM 当前不支持
sp = SamplingParams(teacher_forcing=True, target_tokens=gt_token_ids)
outputs = llm.generate(prompts, sp)
# outputs[i].logprob 直接返回 GT 的 log-likelihood
```

**优点**：
- 最高效（利用 vLLM 的 batching 和 KV cache）
- 最干净（不需要单独的 HF 模型）

**缺点**：
- 需要修改 vLLM 源码（涉及 scheduler、sampler、worker）
- 仍然需要 GT trace

**代码改动清单**：
1. Fork vLLM，在 `SamplingParams` 加 `teacher_forcing` 和 `target_tokens` 参数
2. 修改 `sampler.py`，当 `teacher_forcing=True` 时，不采样，直接用 `target_tokens`
3. 在 `RequestOutput` 里返回累积的 logprob
4. （剩余同方案 B）

---

## 推荐方案

### 如果**没有 GT trace**（当前情况）

→ **方案 A（混合 reward）**，最容易实现，能立刻提供连续信号，虽然不是纯 log-likelihood。

### 如果能**生成 GT trace**

有两种方法：

#### 1. 用符号求解器（最可靠）
```python
from itertools import permutations, product

def solve_countdown(numbers, target):
    """穷举所有可能的表达式，返回第一个正确的"""
    ops = ['+', '-', '*', '/']
    for perm in permutations(numbers):
        for op_combo in product(ops, repeat=len(numbers)-1):
            expr = f"{perm[0]}"
            for i, op in enumerate(op_combo):
                expr += f" {op} {perm[i+1]}"
            try:
                if abs(eval(expr) - target) < 1e-5:
                    return expr
            except:
                continue
    return None  # 无解
```

这样可以为数据集里的每个问题生成一个 GT trace（虽然可能不是最"人类化"的推理）。

#### 2. 用更强的模型生成（如 GPT-4）
让 GPT-4 生成推理过程，人工检查一部分，作为 pseudo-GT。

然后用 **方案 B**（HF forward pass）。

---

## 总结：要添加的逻辑

### 最小实现（方案 A，无需 GT）

```diff
# es_train_axis.py

+parser.add_argument("--logprob_reward", action="store_true",
+                    help="Use log-likelihood as continuous reward (no GT needed)")
+parser.add_argument("--logprob_weight", type=float, default=0.01,
+                    help="Weight for logprob term in reward = correctness + w*logprob")

-sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42, stop=C.STOP)
+sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42, stop=C.STOP,
+                    logprobs=1 if a.logprob_reward else None)

 outputs = llm.generate(prompts, sp)
 for i, o in enumerate(outputs):
     text = o.outputs[0].text
+    if a.logprob_reward:
+        token_logprobs = [list(lp.values())[0].logprob for lp in o.outputs[0].logprobs]
+        total_logprob = sum(token_logprobs)
+        fitness[i] = row_reward(text, batch[i]) + a.logprob_weight * total_logprob
+    else:
+        fitness[i] = row_reward(text, batch[i])
-    fitness[i] = row_reward(text, batch[i])
```

### 完整实现（方案 B，需要 GT）

需要额外：
1. **GT trace 数据文件**（如 `countdown_gt_traces.json`）
2. **HF 模型加载**（与 vLLM 并行，或共享权重）
3. **`compute_gt_loglikelihood()` 函数**
4. **新的 reward 函数**，用 GT logprob 代替 correctness

---

## 实际建议

**对于你们当前的实验**，我建议先试 **方案 A（混合 reward，无需 GT）**：
- 改动最小（~20 行代码）
- 立刻可用
- 能验证"连续 reward 是否真能提升 ES"这个假设
- 如果有效，再投入精力生成 GT traces 做方案 B

要我实现方案 A 吗？
