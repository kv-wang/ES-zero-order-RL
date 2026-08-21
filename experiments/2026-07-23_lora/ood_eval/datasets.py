from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from datasets import load_dataset


@dataclass
class Example:
    question: str
    answer: str
    metadata: Dict


def _as_examples(rows: Iterable[dict], question_key: str, answer_key: str, dataset_name: str) -> List[Example]:
    out = []
    for r in rows:
        out.append(
            Example(
                question=str(r[question_key]),
                answer=str(r[answer_key]),
                metadata={"dataset": dataset_name},
            )
        )
    return out


def load_math_split(split: str):
    ds = load_dataset("hendrycks/competition_math", split=split)
    return _as_examples(ds, "problem", "solution", "math")


def load_gsm8k_split(split: str):
    ds = load_dataset("gsm8k", "main", split=split)
    return _as_examples(ds, "question", "answer", "gsm8k")


def load_eval_dataset(name: str) -> List[Example]:
    # Map dataset names to HF datasets/adapters.
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return _as_examples(ds, "problem", "answer", "math500")
    if name == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="test")
        return _as_examples(ds, "Body", "Answer", "svamp")
    if name == "asdiv":
        ds = load_dataset("EleutherAI/asdiv", split="validation")
        return _as_examples(ds, "question", "answer", "asdiv")
    if name == "gsm-hard":
        ds = load_dataset("reasoning-machines/gsm-hard", split="test")
        return _as_examples(ds, "input", "target", "gsm-hard")
    if name == "aime2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return _as_examples(ds, "problem", "answer", "aime2024")
    if name == "amc":
        ds = load_dataset("math-ai/amc", split="test")
        return _as_examples(ds, "problem", "answer", "amc")
    if name == "minerva":
        ds = load_dataset("AI-MO/Minerva_Math", split="test")
        return _as_examples(ds, "problem", "answer", "minerva")
    if name == "olympiadbench":
        ds = load_dataset("Hothan/OlympiadBench", split="test")
        return _as_examples(ds, "question", "final_answer", "olympiadbench")
    raise ValueError(f"Unknown eval dataset: {name}")


def build_source_splits(train_dataset: str, train_size: int, val_size: int, seed: int):
    if train_dataset == "math":
        all_train = load_math_split("train")
    elif train_dataset == "gsm8k":
        all_train = load_gsm8k_split("train")
    else:
        raise ValueError(f"Unsupported train_dataset={train_dataset}")

    import random

    rng = random.Random(seed)
    idx = list(range(len(all_train)))
    rng.shuffle(idx)
    chosen = idx[: train_size + val_size]
    train_idx = chosen[:train_size]
    val_idx = chosen[train_size : train_size + val_size]

    train = [all_train[i] for i in train_idx]
    val = [all_train[i] for i in val_idx]
    return train, val
