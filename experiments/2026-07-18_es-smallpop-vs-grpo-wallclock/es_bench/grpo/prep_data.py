#!/usr/bin/env python
"""Build verl-format train/val parquet from the SAME MATH levels 3-5 split and
data_seed the ES trainer uses, so GRPO and ES see the same problem distribution.

verl RLHFDataset row schema:
  data_source, prompt=[{role,content}], ability, reward_model={style,ground_truth}, extra_info
The chat template is applied by verl using the model tokenizer (same INSTRUCTION
as the ES trainer's build_prompt), keeping prompts identical across methods.
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from es_bench.data_math import make_split, INSTRUCTION


def to_rows(split, name, source="math_es"):
    rows = []
    for i, ex in enumerate(split):
        rows.append({
            "data_source": source,
            "prompt": [{"role": "user", "content": ex["question"] + "\n\n" + INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(ex["gt"])},
            "extra_info": {"index": i, "split": name, "uid": i, "level": ex.get("level")},
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="math")
    p.add_argument("--levels", default="3,4,5")
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--out_dir", default="es_bench/grpo/data")
    a = p.parse_args()
    levels = [int(x) for x in a.levels.split(",") if x.strip()] or None
    train, val = make_split(a.dataset, a.train_size, a.val_size, a.data_seed, levels=levels)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(to_rows(train, "train")).to_parquet(os.path.join(a.out_dir, "train.parquet"))
    pd.DataFrame(to_rows(val, "val")).to_parquet(os.path.join(a.out_dir, "val.parquet"))
    print(f"wrote {len(train)} train, {len(val)} val to {a.out_dir}")


if __name__ == "__main__":
    main()
