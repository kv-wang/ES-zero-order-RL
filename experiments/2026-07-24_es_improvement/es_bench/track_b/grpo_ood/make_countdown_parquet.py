#!/usr/bin/env python
"""Countdown train parquet for verl GRPO (2026-07-29 battery).

Rows [300:] of countdown.json (the eval battery pins rows [:300], so train/eval are disjoint).
The `prompt` chat list holds the dataset's raw `context` as the single user message; it is
served through an IDENTITY chat template (make_raw_template_model.py), so the string verl
actually rolls out is char-identical to the ES arms' training prompt and to
eval_core._eval_countdown. numbers/target ride in extra_info for reward_countdown.py.
"""
import json, os, sys

import pandas as pd

SRC = "/home/hyin66/es-fine-tuning-paper/countdown/data/countdown.json"
EVAL_RESERVED = 300


def rows(split, data, base):
    return [{
        "data_source": "countdown_es",
        "prompt": [{"role": "user", "content": r["context"]}],
        "ability": "countdown",
        "reward_model": {"style": "rule", "ground_truth": str(r["target"])},
        # target stays the VERBATIM source string (43 rows have float-string targets;
        # answer_reward_function compares via float(target))
        "extra_info": {"index": base + i, "uid": base + i, "split": split,
                       "numbers": [int(x) for x in r["numbers"]], "target": str(r["target"])},
    } for i, r in enumerate(data)]


def main(out_dir):
    data = json.load(open(SRC))
    train, val = data[EVAL_RESERVED:], data[:EVAL_RESERVED]
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows("train", train, EVAL_RESERVED)).to_parquet(os.path.join(out_dir, "train.parquet"))
    pd.DataFrame(rows("val", val, 0)).to_parquet(os.path.join(out_dir, "val.parquet"))
    print(f"wrote {len(train)} train / {len(val)} val rows to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
