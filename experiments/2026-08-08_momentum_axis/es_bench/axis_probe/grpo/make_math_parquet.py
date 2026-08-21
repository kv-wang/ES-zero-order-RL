#!/usr/bin/env python
"""MATH L3-5 train/val parquet for verl GRPO, built from the SAME call the ES trainer makes.

Comparability is the whole point of this file. The ES arms of this try build their pool with

    data_math.make_split(C.DATASET, C.TRAIN_SIZE, 200, C.DATA_SEED, levels=C.LEVELS)

so this script calls exactly that, from this try's own es_bench. Two consequences:

* The gold answers come from the POST-FIX brace-counting extractor. The parquet that
  2026-07-24 left behind (track_b/grpo_ood/data_math/, dated 2026-07-25) was built when
  `_math_gt` still used r"\\boxed\\{([^{}]+)\\}", so its pool was drawn from 4180 problems
  instead of 5584 and the TRAIN_SIZE=2000 sample is a DIFFERENT set of problems. It must
  not be reused for a comparison against the 2026-08-10 ES arms.

* The prompt string is built by data_math.build_prompt with this try's tokenizer and chat
  template, so what verl rolls out is character-identical to what the ES members saw.

The reward is not defined here: verl is pointed at es_bench/shared_reward.py::verl_compute_score,
the same object the ES arms optimize (protocol constraint 2).
"""
from __future__ import annotations
import argparse, json, os, sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    TRY = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))      # <try>/
    sys.path.insert(0, TRY)
    sys.path.insert(0, os.path.join(TRY, "es_bench", "axis_probe", "src"))
    import config as C
    from es_bench import data_math
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.MODEL)
    train, val = data_math.make_split(C.DATASET, C.TRAIN_SIZE, 200, C.DATA_SEED, levels=C.LEVELS)

    def rows(split, data, base):
        return [{
            "data_source": "math_es",
            # build_prompt already applies the chat template, so hand verl the finished
            # string as a single user turn rather than letting it re-template.
            "prompt": [{"role": "user", "content": data_math.build_prompt(tok, r["question"])}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(r["gt"])},
            "extra_info": {"index": base + i, "uid": base + i, "split": split,
                           "level": r.get("level")},
        } for i, r in enumerate(data)]

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows("train", train, 0)).to_parquet(os.path.join(a.out_dir, "train.parquet"))
    pd.DataFrame(rows("val", val, len(train))).to_parquet(os.path.join(a.out_dir, "val.parquet"))

    # Provenance stamp. The 2026-07-25 parquet was silently reused for weeks after the extractor
    # it was built with turned out to be broken; a bare "file exists" check cannot catch that.
    # run_grpo.sh verifies this stamp before skipping the build.
    import hashlib
    ae = os.path.join(TRY, "ood_eval", "answer_extraction.py")
    prov = {
        "built_by": os.path.basename(__file__),
        "model": C.MODEL, "dataset": C.DATASET, "levels": list(C.LEVELS),
        "train_size": C.TRAIN_SIZE, "data_seed": C.DATA_SEED,
        "n_train": len(train), "n_val": len(val),
        "answer_extraction_md5": hashlib.md5(open(ae, "rb").read()).hexdigest(),
        "braced_gold_frac": round(sum(1 for r in train if "{" in str(r["gt"])) / len(train), 4),
    }
    with open(os.path.join(a.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("  provenance:", json.dumps(prov))
    print(f"wrote {len(train)} train / {len(val)} val rows to {a.out_dir}")
    print(f"  model={C.MODEL}  dataset={C.DATASET} levels={C.LEVELS} "
          f"train_size={C.TRAIN_SIZE} data_seed={C.DATA_SEED}")
    print(f"  first gold: {train[0]['gt']!r}   (post-fix extractor keeps braced answers)")


if __name__ == "__main__":
    main()
