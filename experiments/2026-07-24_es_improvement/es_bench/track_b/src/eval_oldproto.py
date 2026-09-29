#!/usr/bin/env python
"""Evaluate a model (optionally + an in-memory LoRA theta) on the OLD-protocol battery.

Old protocol = eval_core.eval_on_llm(cap=300, max_tokens=2048), i.e. exactly what produced
results/exp3b_math/{base_3b.json, grpo_full_3b.json} and OOD_COMPARISON_MATH.csv. The overnight
battery used max_tokens=512, which truncates 50-80% of MATH-family responses and is therefore
not comparable to those rows.

With --theta, the base row (no adapter) and the adapter row are produced in the SAME process,
so greedy batch-sensitivity cannot contaminate the within-run delta.
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--theta", default=None, help="LoRA theta .pt from es_lora_main (optional)")
    ap.add_argument("--lambdas", default="1.0", help="comma list of shrinkage scales for --theta")
    ap.add_argument("--eval_cap", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=None, help="adapter rank of the theta ckpt")
    ap.add_argument("--targets", default=None, help="comma list; must match the theta ckpt")
    ap.add_argument("--perq", action="store_true",
                    help="write per-question jsonl next to --out (row order is deterministic, "
                         "so rows pair across arms by index for McNemar)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
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
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "track_a", "src"))

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    import eval_core
    import lora_cfg
    import config

    tok = AutoTokenizer.from_pretrained(a.model)
    use_lora = a.theta is not None
    rank = a.rank if a.rank is not None else lora_cfg.LORA_RANK
    targets = a.targets.split(",") if a.targets else lora_cfg.TARGET_MODULES
    kw = {}
    if use_lora:
        kw = dict(enable_lora=True, max_loras=2, max_lora_rank=rank, max_cpu_loras=8,
                  worker_extension_cls="es_bench.track_b.src.lora_main_worker.LoRAMainWorker")
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=config.EVAL_GPU_MEM_UTIL,
              max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True, **kw)

    results = {"model": a.model, "theta": a.theta,
               "protocol": {"cap": a.eval_cap, "max_tokens": a.max_tokens, "greedy": True,
                            "note": "old protocol, matches base_3b.json / grpo_full_3b.json"},
               "entries": {}}

    def flush():
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)

    def pfx(tag):
        return (os.path.splitext(a.out)[0] + "__" + tag) if a.perq else None

    # row 0: the loaded model with no adapter
    ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=a.max_tokens,
                               perq_prefix=pfx(a.tag))
    results["entries"][a.tag] = ev
    flush()
    print(f"[eval] {a.tag}: " + ", ".join(f"{k}={v['accuracy']}" for k, v in ev.items()), flush=True)

    if use_lora:
        llm.collective_rpc("es_lora_init",
                           args=(lora_cfg.build_template(a.model, rank=rank, targets=targets),
                                 rank, lora_cfg.LORA_ALPHA, targets, 0))
        llm.collective_rpc("es_lora_load_theta", args=(torch.load(a.theta, map_location="cpu"),))
        for i, lam in enumerate([float(x) for x in a.lambdas.split(",")]):
            lid = 800001 + i
            llm.collective_rpc("es_lora_inject_scaled", args=(lid, lam))
            req = LoRARequest(f"theta_l{lam}", lid, "/es/inmem")
            key = f"{a.tag}_lambda{lam:g}"
            ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=a.max_tokens,
                                       lora_request=req, perq_prefix=pfx(key))
            results["entries"][key] = ev
            flush()
            print(f"[eval] {key}: " + ", ".join(f"{k}={v['accuracy']}" for k, v in ev.items()),
                  flush=True)

    flush()
    print(f"[eval] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
