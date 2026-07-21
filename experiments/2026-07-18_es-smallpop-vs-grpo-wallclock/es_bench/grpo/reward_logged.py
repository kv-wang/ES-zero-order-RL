"""verl custom reward = the SAME shared math_reward, plus per-call logging so we
can reconstruct the GRPO 'zero-advantage group' rate offline.

verl's NaiveRewardManager calls compute_score(...) once per sample in the driver
process. We append {gidx, uid, level, reward} to $ES_GRPO_REWARD_LOG. Offline,
group by step (gidx // samples_per_step) then by uid; a group whose rewards are
all identical => zero-advantage group (GRPO advantage 0 for that prompt).
"""
from __future__ import annotations
import json, os, sys, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from es_bench.shared_reward import math_reward

_LOG_PATH = os.environ.get("ES_GRPO_REWARD_LOG", "")
_lock = threading.Lock()
_counter = {"i": 0}
_fh = open(_LOG_PATH, "a") if _LOG_PATH else None


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    r = math_reward(solution_str or "", ground_truth or "")["reward"]
    if _fh is not None:
        with _lock:
            gidx = _counter["i"]; _counter["i"] += 1
            uid = level = split = None
            if isinstance(extra_info, dict):
                uid = extra_info.get("uid")
                level = extra_info.get("level")
                split = extra_info.get("split")
            _fh.write(json.dumps({"gidx": gidx, "uid": uid, "level": level,
                                  "split": split, "reward": r}) + "\n")
            _fh.flush()
    return r
