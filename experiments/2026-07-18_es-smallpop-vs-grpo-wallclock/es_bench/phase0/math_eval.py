#!/usr/bin/env python
"""Phase 0 base-accuracy probe on MATH-500, with per-level breakdown, using
vLLM greedy pass@1 and the SHARED reward (es_bench.shared_reward.math_reward).
Used to decide the Phase 1 training task when GSM8K is too easy (acc>0.7)."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--levels", default="3,4,5")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench.data_math import build_prompt

    levels = {int(x) for x in a.levels.split(",") if x.strip()}
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [r for r in ds if int(r["level"]) in levels][: a.n]

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [build_prompt(tok, r["problem"]) for r in rows]

    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    gen_s = time.perf_counter() - t0

    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    per_level = defaultdict(lambda: [0, 0])
    n_correct = 0
    with open(a.out_prefix + "_perq.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            rew = math_reward(text, r["answer"])["reward"]
            c = int(rew >= 1.0)
            n_correct += c
            lvl = int(r["level"])
            per_level[lvl][0] += c
            per_level[lvl][1] += 1
            f.write(json.dumps({"level": lvl, "gt": r["answer"], "correct": c,
                                "out_tokens": len(o.outputs[0].token_ids)}) + "\n")
    n = len(rows)
    summary = {
        "model": a.model, "dataset": "MATH-500", "levels": sorted(levels), "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "per_level_accuracy": {str(k): round(v[0] / v[1], 4) for k, v in sorted(per_level.items())},
        "per_level_n": {str(k): v[1] for k, v in sorted(per_level.items())},
        "gen_seconds": round(gen_s, 2), "max_tokens": a.max_tokens,
    }
    with open(a.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
