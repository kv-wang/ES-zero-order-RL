"""Countdown as a TRAINING task (2026-07-29 battery: train countdown, eval OOD math).

eval_core._eval_countdown pins its probe to countdown.json[:cap] with cap<=300, so the
train pool is rows [300:] -- disjoint from every eval row by construction. Prompts are the
dataset's own raw `context` (completion-style, opens with <think>), NO chat template --
matching the original ES setup and the eval protocol. Reward is the same binary
countdown_task.answer_reward_function used by eval_core and the GRPO arm.
"""
from __future__ import annotations
import json, sys

COUNTDOWN_JSON = "/home/hyin66/es-fine-tuning-paper/countdown/data/countdown.json"
COUNTDOWN_TASK_DIR = "/home/hyin66/es-fine-tuning-paper/countdown"
EVAL_RESERVED = 300     # eval_core._eval_countdown reads data[:cap]; cap is 300 everywhere


def load_split():
    data = json.load(open(COUNTDOWN_JSON))
    return data[EVAL_RESERVED:], data[:EVAL_RESERVED]   # train (1900), val (== the eval slice)


def reward(text: str, row: dict) -> float:
    if COUNTDOWN_TASK_DIR not in sys.path:
        sys.path.insert(0, COUNTDOWN_TASK_DIR)
    from countdown_task import answer_reward_function
    return float(answer_reward_function(text, row["numbers"], row["target"]))
