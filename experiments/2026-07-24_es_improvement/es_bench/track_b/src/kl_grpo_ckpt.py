#!/usr/bin/env python
"""KL-to-base for a GRPO checkpoint, on the SAME proxy the ES arms report.

The ES trainers measure KL in-process (capture base-greedy completions at step 0, then
score them teacher-forced under theta_final). GRPO's final weights live in a separate HF
directory, so the same measurement needs two vLLM instances and two invocations:

  --mode capture  --model <BASE>            --rec rec.json    # base greedy + base logprobs
  --mode measure  --model <MERGED_ACTOR>    --rec rec.json    # NLL drift of those tokens

Prompt set is reconstructed byte-identically to es_train_fullparam.py:
  make_split("math", TRAIN_SIZE=2000, 300, DATA_SEED=1234, levels=[3,4,5]) -> val[:200],
  build_prompt(), greedy, max_tokens=512.
This makes the resulting D directly comparable to the `kl_proxy_drift` in the ES summaries.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["capture", "measure"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--rec", required=True)
    p.add_argument("--out", default=None, help="measure mode: json to write")
    p.add_argument("--tag", default="grpo")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
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
    from vllm import LLM
    from es_bench import data_math
    import kl as klmod

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    _, val = data_math.make_split("math", 2000, 300, a.data_seed, levels=[3, 4, 5])
    prompts = [data_math.build_prompt(tok, v["question"]) for v in val[:a.n_prompts]]

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=2048, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False)

    if a.mode == "capture":
        rec = klmod.capture_base(llm, prompts, max_tokens=a.max_tokens)
        self_drift = klmod.drift(llm, tok, rec)
        Path(a.rec).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model": a.model, "n": len(rec), "self_drift": self_drift, "rec": rec},
                  open(a.rec, "w"))
        print(f"[capture] {len(rec)} completions from {a.model}; "
              f"base-vs-base self-drift={self_drift:.3e} (expect ~0)", flush=True)
    else:
        blob = json.load(open(a.rec))
        rec = blob["rec"]
        d = klmod.drift(llm, tok, rec)
        res = {"tag": a.tag, "target_model": a.model, "base_model": blob["model"],
               "n_prompts": len(rec), "max_tokens": a.max_tokens,
               "base_self_drift": blob["self_drift"], "kl_proxy_drift": d}
        print(f"[measure] {a.tag}: KL proxy drift D = {d:.6f}  "
              f"(base self-drift {blob['self_drift']:.3e})", flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=2)
            print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
