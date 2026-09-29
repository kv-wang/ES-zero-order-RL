"""verl custom reward for countdown = the SAME binary countdown_task.answer_reward_function
used by the ES trainers and eval_core, plus the per-call logging of grpo/reward_logged.py
(so zero-advantage-group rates stay reconstructible offline)."""
from __future__ import annotations
import json, os, sys, threading

sys.path.insert(0, "/home/hyin66/es-fine-tuning-paper/countdown")
from countdown_task import answer_reward_function

_LOG_PATH = os.environ.get("ES_GRPO_REWARD_LOG", "")
_lock = threading.Lock()
_counter = {"i": 0}
_fh = open(_LOG_PATH, "a") if _LOG_PATH else None


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    numbers = [int(x) for x in extra_info["numbers"]]
    r = float(answer_reward_function(solution_str or "", numbers, extra_info["target"]))
    if _fh is not None:
        with _lock:
            gidx = _counter["i"]; _counter["i"] += 1
            _fh.write(json.dumps({"gidx": gidx, "uid": extra_info.get("uid"),
                                  "split": extra_info.get("split"), "reward": r}) + "\n")
            _fh.flush()
    return r
