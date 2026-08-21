"""Math training/eval data as (question, canonical_gt) pairs.

Shared by the ES trainer and the GRPO (verl) data prep so both see the exact
same problems in the same order (data-order seed held constant across configs).
"""
from __future__ import annotations
import importlib.util
import os
import random
import re
from typing import List, Dict, Optional

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")

_GSM_GT = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")

# Gold extraction shares ONE implementation with the reward/eval path so the training
# pool and the scorer can never disagree about what a \boxed answer is. Loaded by file
# path, mirroring shared_reward.py, so ood_eval/datasets.py never shadows HF `datasets`.
#
# Fixed 2026-08-09: this module used to carry its own copy of
#     _BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
# whose character class forbids braces, so `_math_gt` returned None for every nested
# answer and `load_pool` then silently dropped the problem. Measured on MATH L3-5 that
# discarded 1404 of 5586 problems (25.1%), biased toward symbolic answers. See
# experiments/EXTRACTOR_BUG_REPORT.md.
_AE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ood_eval", "answer_extraction.py"
)
_spec = importlib.util.spec_from_file_location("es_answer_extraction_datamath", _AE_PATH)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)
_last_boxed = _ae._last_boxed


def _gsm_gt(answer_field: str) -> Optional[str]:
    m = _GSM_GT.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def _math_gt(solution_field: str) -> Optional[str]:
    content = _last_boxed(solution_field)
    return content.strip() if content is not None else None


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

# Four-shot exemplars for BASE (non instruction-tuned) models. Measured on
# Qwen2.5-Math-1.5B with the zero-shot ChatML prompt below: it does not follow the chat
# format at all -- it echoes the question and loops, 82% of samples run to the 512-token
# training cap, and the answer is unextractable on 27.5% of MATH-500 and 65% of SVAMP.
# Those accuracies measure format compliance, not math. A base model needs the answer
# format demonstrated rather than described, which is also how Qwen's own report and the
# ES paper evaluate base models. Kept deliberately short: the prompt is prepended to every
# one of the 25,600 generations per arm, so exemplar length is a direct budget cost.
FEWSHOT = [
    ("What is $2 + 2 \\times 3$?",
     "Multiplication comes before addition, so $2 \\times 3 = 6$ and $2 + 6 = 8$.\n"
     "The answer is $\\boxed{8}$."),
    ("If $3x - 7 = 8$, what is $x$?",
     "Add 7 to both sides: $3x = 15$. Divide by 3: $x = 5$.\n"
     "The answer is $\\boxed{5}$."),
    ("What is the area of a circle with radius 3?",
     "The area is $\\pi r^2 = \\pi \\cdot 3^2 = 9\\pi$.\n"
     "The answer is $\\boxed{9\\pi}$."),
    ("Simplify $\\frac{4}{8}$.",
     "Both numerator and denominator are divisible by 4, giving $\\frac{1}{2}$.\n"
     "The answer is $\\boxed{\\frac{1}{2}}$."),
]


def build_prompt(tokenizer, question: str, fewshot: bool = None) -> str:
    """Prompt for one question.

    `fewshot=None` (the default) decides from the tokenizer: a model whose eos is
    <|im_end|> is chat-tuned and gets the original zero-shot ChatML prompt, byte-identical
    to every run before 2026-08-14. Anything else is treated as a base model and gets the
    plain-text few-shot format instead -- ChatML on a base model is what produced the
    unextractable-output failure documented above.
    """
    if fewshot is None:
        fewshot = getattr(tokenizer, "eos_token", None) != "<|im_end|>"
    if not fewshot:
        msgs = [{"role": "user", "content": question + "\n\n" + INSTRUCTION}]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    parts = [INSTRUCTION, ""]
    for q, a in FEWSHOT:
        parts += [f"Problem: {q}", f"Solution: {a}", ""]
    parts += [f"Problem: {question}", "Solution:"]
    return "\n".join(parts)
