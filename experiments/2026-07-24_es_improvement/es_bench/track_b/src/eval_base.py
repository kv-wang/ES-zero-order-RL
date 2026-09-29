#!/usr/bin/env python
"""Eval an untrained base model on the OOD/ID battery (anchor for all comparisons)."""
import argparse, json, os, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval_cap", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE)))); sys.path.insert(0, HERE)
    from transformers import AutoTokenizer
    from vllm import LLM
    import eval_core
    import config as C
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=C.EVAL_GPU_MEM_UTIL, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
    ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=2048)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"model": a.model, "eval": ev}, open(a.out, "w"), indent=2)
    print("BASE_EVAL:", json.dumps(ev, indent=2), flush=True)
if __name__ == "__main__":
    main()
