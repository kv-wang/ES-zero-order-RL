#!/usr/bin/env python
"""Untrained-base reference for this try: the anchor every trained arm is measured against.

Same code path as every arm here -- eval_core.eval_on_llm, cap 300,
max_tokens=training cap (config.MAX_TOKENS unless --max_tokens), fp16,
greedy, the six pinned sets, the post-fix brace-counting extractor -- and the same
`*_summary.json` / `*_perq.jsonl` layout, so compare_all.py picks it up without special-casing
and paired McNemar against any arm works directly.

No training, so no KL row: the KL proxy is defined as drift FROM this model.
"""
from __future__ import annotations
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--eval_cap", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="completion cap; default config.MAX_TOKENS (same as training)")
    # The countdown line needs the untrained reference measured on countdown too -- that is the
    # ID metric there, and every trained arm is compared against it.
    ap.add_argument("--include_countdown", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, os.path.join(HERE, "src"))

    from transformers import AutoTokenizer
    from vllm import LLM
    import eval_core
    import config as C

    tok = AutoTokenizer.from_pretrained(C.MODEL)
    llm = LLM(model=C.MODEL, dtype=C.DTYPE, gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
    t0 = time.perf_counter()
    eval_max_tokens = a.max_tokens if a.max_tokens is not None else C.MAX_TOKENS
    ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=eval_max_tokens,
                               perq_prefix=a.out_prefix + "_finaleval",
                               include_countdown=a.include_countdown)
    summary = {
        "variant": "base", "axis": None, "model": C.MODEL,
        "num_steps": 0, "total_generations": 0, "wall_clock_s": 0.0,
        "kl_proxy_drift": None,          # KL is defined as drift from this model
        "eval_max_tokens": eval_max_tokens,
        "eval_final": ev, "eval_seconds": time.perf_counter() - t0,
        "population_size": None, "pop_seed": None, "pair_tie_rate": None,
        "final_cum_disp": None, "zero_update_rate": None, "s_per_step_mean": None,
    }
    with open(a.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("EVAL_FINAL:", json.dumps(ev, indent=2), flush=True)


if __name__ == "__main__":
    main()
