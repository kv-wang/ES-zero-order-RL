# ES 连续 Reward 实现

## 动机

ES 的梯度估计依赖 fitness 的 z-score 分布。Countdown 任务原本使用纯 0/1 二值 reward（答案完全正确得 1，其他情况得 0），导致：
- **低分辨率**：batch=100 时 fitness 只有 101 档（0.00, 0.01, ..., 1.00）
- **高 tie rate**：在小 batch（如 B=8）下，约 67% 的 member pairs 完全同分
- **梯度信号稀疏**：当大部分 member 都答错时，fitness_std 很小，z-score 分辨率低

虽然 ES 只能拿到 forward pass（vLLM generation）+ reward，看不到 logits，但可以通过**给格式打分**来提供连续信号。

## 实现

### 1. Reward 函数（已存在于 countdown_task.py）

```python
def reward_function(response, numbers, target, end_token):
    """Total reward = 0.1 × format_reward + answer_reward"""
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, numbers, target)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {"format_reward": format_reward, "answer_reward": answer_reward}
    }
```

**format_reward** ∈ [0, 1]：
- 1.0 — 完整格式 `<think>...</think>\n<answer>...</answer>`
- 0.6 — 有 `<answer>` 但无 `<think>`（0.5 + 0.1）
- 0.1 — 有 `<think>` 但无 `<answer>`
- 0.0 — 两个标签都没有

**answer_reward** ∈ {0, 1}：
- 1.0 — 最后一个 `<answer>` 里的表达式用了所有数字恰好一次且 eval 结果等于 target
- 0.0 — 其他所有情况

**总 reward** ∈ [0, 1.1]：
- 1.1 — 完美（正确答案 + 完整格式）
- 1.05 — 正确但只有 `<answer>` 标签
- 0.1 — 错误但有完整格式（format credit）
- 0.0 — 错误且无格式

### 2. data_countdown.py 接口

```python
def reward(text: str, row: dict) -> float:
    """Binary reward: 0/1 based on answer correctness only."""
    return float(answer_reward_function(text, row["numbers"], row["target"]))

def reward_continuous(text: str, row: dict) -> float:
    """Continuous reward: 0.1×format + answer ∈ [0, 1.1]."""
    return float(reward_function(text, row["numbers"], row["target"], end_token=None)["reward"])
```

### 3. es_train_axis.py flag

```bash
--continuous_reward    # 启用连续 reward（仅对 countdown 有效）
```

训练器会根据这个 flag 选择 `dc.reward` 还是 `dc.reward_continuous`，并在 summary JSON 里记录 `"continuous_reward": true/false`。

## 使用方法

### 命令行

```bash
# 原有的二值 reward（默认）
python axis_probe/src/es_train_axis.py \
  --dataset countdown --variant tilt \
  --population_size 30 --batch 100 --num_steps 100 \
  --sigma 1e-3 --alpha 5e-4 \
  --tilt_kappa 0.2 --tilt_mom_beta 0.9 --tilt_warmup 1 \
  --out_prefix results/tilt_binary

# 启用连续 reward
python axis_probe/src/es_train_axis.py \
  --dataset countdown --variant tilt \
  --population_size 30 --batch 100 --num_steps 100 \
  --sigma 1e-3 --alpha 5e-4 \
  --tilt_kappa 0.2 --tilt_mom_beta 0.9 --tilt_warmup 1 \
  --continuous_reward \
  --out_prefix results/tilt_continuous
```

### 脚本集成

修改 `run_countdown_3b.sh` 添加：

```bash
CONTINUOUS_REWARD="${CONTINUOUS_REWARD:-0}"  # 默认关闭

# 在 es_train_axis.py 调用中添加：
CONT_FLAG=""
[[ "$CONTINUOUS_REWARD" == "1" ]] && CONT_FLAG="--continuous_reward"

"$P4_PY" axis_probe/src/es_train_axis.py \
  --dataset countdown --variant tilt \
  ... \
  $CONT_FLAG \
  --out_prefix "$RES/${prefix}"
```

然后运行：
```bash
CONTINUOUS_REWARD=1 ./run_countdown_3b.sh
```

## 预期效果

### 优点
- **更细的梯度信号**：即使答案全错，format 好的 member（fitness ~0.1）仍能和完全乱输出的（fitness ~0.0）区分开
- **可能提升 extract_rate**：模型学会"先写标签再答题"可以减少截断（当前 extract_rate ~0.5，base 模型 0.89）
- **减少 zero-update 风险**：虽然当前 B=100 下已经没有 zero-update，但在小 batch 或早期 step 时连续 reward 能保持 fitness_std > 0

### 局限
- **信号仍然弱**：format_reward 的权重只有 0.1，对于完全答错的 member，fitness 从 0.0 变成 0.1 的提升很小
- **不改变"解题能力"的分辨率**：answer_reward 仍然是 0/1，连续 reward 只是在"答错"这个大类里加了细分
- **横向比较要统一**：如果测试连续 reward，vanilla/baseaxis/tilt 都得同时用，否则无法比较方法优劣

## 验证

运行测试脚本：
```bash
/home/hyin66/micromamba/envs/verl/bin/python - <<'PY'
import sys
sys.path.insert(0, 'experiments/2026-08-08_momentum_axis/es_bench')
sys.path.insert(0, 'countdown')
from data_countdown import reward, reward_continuous

row = {"numbers": [3, 5, 25], "target": 200}
text_correct = '<think>ok</think>\n<answer>(5+3)*25</answer>'
text_wrong = '<think>ok</think>\n<answer>5+3+25</answer>'

print(f"Correct: binary={reward(text_correct, row):.2f}, continuous={reward_continuous(text_correct, row):.2f}")
print(f"Wrong:   binary={reward(text_wrong, row):.2f}, continuous={reward_continuous(text_wrong, row):.2f}")
PY
```

预期输出：
```
Correct: binary=1.00, continuous=1.10
Wrong:   binary=0.00, continuous=0.10
```

## 后续改进方向

如果 format credit (0.1) 太小，可以考虑：
1. **调整权重**：改成 `0.2 × format + answer`（需要同时调整学习率 α）
2. **Partial credit**：检查表达式里用了几个正确的数字，给 0.0 ~ 1.0 的连续分
3. **Length penalty**：`reward - λ·(len/max_len)` 鼓励简洁，可能提升 extract_rate
4. **增大 N**：当前 N=30，可以试 50/100 来直接提升梯度估计的方差，比改 reward 更直接

## 文件清单

- `countdown/countdown_task.py` — `reward_function`、`answer_reward_function`、`format_reward_function`（已存在）
- `experiments/2026-08-08_momentum_axis/es_bench/data_countdown.py` — `reward`、`reward_continuous`（新增）
- `experiments/2026-08-08_momentum_axis/es_bench/axis_probe/src/es_train_axis.py` — `--continuous_reward` flag（新增）
