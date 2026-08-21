"""Single shared reward object imported by BOTH the ES trainer and the GRPO
(verl) control, so the two methods optimize the identical scalar reward
(protocol constraint 2).

Reward = 1.0 if the model's extracted final answer matches the ground-truth
canonical answer, else 0.0. Extraction reuses ood_eval/answer_extraction.

`gt` is expected to be the CANONICAL answer already (a bare number/string for
GSM8K, the boxed content for MATH) — the data layer normalizes it up front so
the reward stays a pure string->scalar function.
"""
from __future__ import annotations
import importlib.util
import os
import re
from typing import Any, Dict

# Load ood_eval/answer_extraction.py by file path WITHOUT touching sys.path,
# so we don't shadow the HF `datasets` package with ood_eval/datasets.py.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AE_PATH = os.path.join(_REPO, "ood_eval", "answer_extraction.py")
_spec = importlib.util.spec_from_file_location("es_answer_extraction", _AE_PATH)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)
extract_final_answer = _ae.extract_final_answer
normalize_answer = _ae.normalize_answer

_NUM = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _num(s: str):
    s = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def math_reward(response: str, gt: str) -> Dict[str, Any]:
    """Return {'reward': 0.0/1.0, 'reward_info': {...}} for a math answer."""
    ext = extract_final_answer(response)
    correct = 0.0
    if ext.success:
        gt_n, pred_n = _num(gt), _num(ext.extracted)
        if gt_n is not None and pred_n is not None:
            correct = float(abs(gt_n - pred_n) < 1e-4)
        else:  # non-numeric answers (some MATH): fall back to string match
            correct = float(normalize_answer(str(ext.extracted)) == normalize_answer(str(gt)))
    return {
        "reward": correct,
        "reward_info": {
            "extracted": bool(ext.success),
            "used_boxed": bool(ext.used_boxed),
        },
    }


# verl-style adapter: verl custom reward fns take (data_source, solution_str,
# ground_truth, extra_info) and return a float. This wraps the SAME object.
def verl_compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    return math_reward(solution_str or "", ground_truth or "")["reward"]


if __name__ == "__main__":
    # tiny self-test
    print(math_reward("The answer is \\boxed{18}.", "18"))
    print(math_reward("So we get 42", "42"))
    print(math_reward("no answer here", "5"))
