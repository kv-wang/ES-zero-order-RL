#!/usr/bin/env python
"""Merge a verl FSDP GRPO checkpoint -> HF, then run the OOD/ID battery in vLLM.
Usage: eval_grpo_ood.py --actor_dir <ckpt>/global_step_N/actor --tag grpo_full --out <json>
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor_dir", required=True)     # verl .../global_step_N/actor
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_cap", type=int, default=300)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--perq", action="store_true",
                    help="write per-question jsonl next to --out (rows pair across arms by index)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, os.path.join(HERE, "..", "src"))

    hf_dir = os.path.join(os.path.dirname(a.actor_dir), "hf_merged")
    if not os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"[merge] {a.actor_dir} -> {hf_dir}", flush=True)
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", a.actor_dir, "--target_dir", hf_dir], check=True)

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    import eval_core
    import config as C
    tok = AutoTokenizer.from_pretrained(hf_dir)
    adapter_dir = os.path.join(hf_dir, "lora_adapter")
    is_lora = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    if is_lora:  # verl saved base + separate LoRA adapter -> load base + apply adapter from disk
        print(f"[eval] LoRA checkpoint: base {hf_dir} + adapter {adapter_dir}", flush=True)
        llm = LLM(model=hf_dir, dtype="float16", gpu_memory_utilization=C.EVAL_GPU_MEM_UTIL,
                  max_model_len=4096,
                  enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
                  enable_lora=True, max_lora_rank=64)
        lreq = LoRARequest("grpo_lora", 1, adapter_dir)
        ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=2048, lora_request=lreq,
                                   perq_prefix=(os.path.splitext(a.out)[0] + "__" + a.tag
                                                if a.perq else None))
    else:
        llm = LLM(model=hf_dir, dtype="float16", gpu_memory_utilization=C.EVAL_GPU_MEM_UTIL,
                  max_model_len=4096,
                  enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
        ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=2048,
                                   perq_prefix=(os.path.splitext(a.out)[0] + "__" + a.tag
                                                if a.perq else None))
    res = {"method": a.tag, "hf_dir": hf_dir, "eval": ev}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("GRPO_OOD:", json.dumps(ev, indent=2), flush=True)


if __name__ == "__main__":
    main()
