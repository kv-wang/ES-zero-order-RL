"""Math training/eval data as (question, canonical_gt) pairs.

Shared by the ES trainer and the GRPO (verl) data prep so both see the exact
same problems in the same order (data-order seed held constant across configs).
"""
from __future__ import annotations
import os
import random
import re
from typing import List, Dict, Optional

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_GSM_GT = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def _gsm_gt(answer_field: str) -> Optional[str]:
    m = _GSM_GT.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def _math_gt(solution_field: str) -> Optional[str]:
    m = _BOXED.findall(solution_field)
    return m[-1].strip() if m else None


def load_pool(dataset: str, levels: Optional[List[int]] = None) -> List[Dict]:
    """Return the full training pool as [{question, gt, level}]."""
    from datasets import load_dataset
    out = []
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for r in ds:
            gt = _gsm_gt(r["answer"])
            if gt is not None:
                out.append({"question": r["question"], "gt": gt, "level": None})
    elif dataset == "math":
        from datasets import concatenate_datasets
        subjects = ["algebra", "counting_and_probability", "geometry",
                    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        ds = concatenate_datasets(
            [load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in subjects])
        for r in ds:
            gt = _math_gt(r["solution"])
            if gt is None:
                continue
            lvl = None
            m = re.search(r"(\d+)", str(r.get("level", "")))
            if m:
                lvl = int(m.group(1))
            if levels and lvl not in levels:
                continue
            out.append({"question": r["problem"], "gt": gt, "level": lvl})
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return out


def make_split(dataset: str, train_size: int, val_size: int, seed: int,
               levels: Optional[List[int]] = None):
    pool = load_pool(dataset, levels=levels)
    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    chosen = idx[: train_size + val_size]
    train = [pool[i] for i in chosen[:train_size]]
    val = [pool[i] for i in chosen[train_size: train_size + val_size]]
    return train, val


INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": question + "\n\n" + INSTRUCTION}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
