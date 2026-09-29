#!/usr/bin/env python
"""Memorization probe: accuracy on the model's OWN training problems vs held-out problems.

GRPO's train reward rose 0.601 -> 0.676 while held-out L3-5 fell below base. If that reward gain
is memorization of the 2000-problem pool (1.6 epochs at bs16x200steps), accuracy ON the pool
should exceed accuracy on exchangeable held-out problems by more than noise. Same split builder
and data_seed as training (data_math.make_split('math', 2000, 300, 42, levels=[3,4,5])), same
greedy cap-2048 protocol as every stored row.

Run once per model (one vLLM instance per process):
  grpo_overfit_probe.py --model Qwen/Qwen2.5-3B-Instruct --out .../base.json
  grpo_overfit_probe.py --model /tmp/.../hf_merged        --out .../grpo.json
"""
from __future__ import annotations
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300, help="problems per side")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--data_seed", type=int, default=42, help="MUST match the GRPO train prep")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, HERE)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench import data_math
    from es_bench.shared_reward import math_reward
    import config

    train, val = data_math.make_split("math", 2000, 300, a.data_seed, levels=[3, 4, 5])
    sides = {"train_pool": train[: a.n], "heldout": val[: a.n]}

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="float16",
              gpu_memory_utilization=config.EVAL_GPU_MEM_UTIL, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    res = {"model": a.model, "n": a.n, "max_tokens": a.max_tokens,
           "data_seed": a.data_seed, "sides": {}}
    for side, rows in sides.items():
        prompts = [data_math.build_prompt(tok, r["question"]) for r in rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        per = [int(math_reward(o.outputs[0].text, r["gt"])["reward"] >= 1.0)
               for o, r in zip(outs, rows)]
        res["sides"][side] = {"accuracy": round(sum(per) / len(per), 4), "n": len(per),
                              "correct": per}
        print(f"[{side}] acc={res['sides'][side]['accuracy']}", flush=True)
    res["train_minus_heldout"] = round(
        res["sides"]["train_pool"]["accuracy"] - res["sides"]["heldout"]["accuracy"], 4)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"train-heldout gap: {res['train_minus_heldout']:+.4f}  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
