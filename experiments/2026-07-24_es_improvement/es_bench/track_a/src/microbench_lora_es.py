#!/usr/bin/env python
"""A0 microbench: LoRA-ES step timing + VRAM for one N. All N members in ONE batched
generate (N adapters x B prompts = N*B requests). Decomposition: adapter-write / gen / update.
cuda-sync + perf_counter, 5 warmup discarded, >=15 timed. Run one N per process (clean VRAM).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--pop_size", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, HERE)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from es_bench import data_math
    import lora_cfg

    N, B = a.pop_size, a.batch
    tmpl = lora_cfg.build_template(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    pool, _ = data_math.make_split("gsm8k", 64, 8, 1234)
    prompts = [data_math.build_prompt(tok, pool[i]["question"]) for i in range(B)]

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=a.gpu_mem_util, max_model_len=2048,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
              enable_lora=True, max_loras=N, max_lora_rank=lora_cfg.LORA_RANK, max_cpu_loras=2 * N + 4,
              worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA, lora_cfg.TARGET_MODULES, 0))
    sp_base = dict(temperature=0.0, max_tokens=a.max_tokens)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    total_steps = a.warmup + a.timed
    for step in range(total_steps):
        seeds = [1 + step * 1000 + i for i in range(N)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = llm.collective_rpc("es_lora_inject_members", args=(1 + step * N, seeds, a.sigma))[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        # N*B requests: member i on all B prompts
        reqs, sps, lrs = [], [], []
        for i, lid in enumerate(ids):
            lr = LoRARequest(f"s{step}m{i}", lid, "/es/inmem")
            for b in range(B):
                reqs.append(prompts[b]); lrs.append(lr)
                sps.append(SamplingParams(seed=seeds[i], **sp_base))
        outs = llm.generate(reqs, sps, lora_request=lrs, use_tqdm=False)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        # dummy fitness (timing only) + real z-score commit
        fit = np.array([sum(len(o.outputs[0].token_ids) for o in outs[i * B:(i + 1) * B]) for i in range(N)], dtype=np.float64)
        z = (fit - fit.mean()) / (fit.std() + 1e-8)
        coeffs = ((5e-4 / N) * z).tolist()
        llm.collective_rpc("es_lora_commit", args=(seeds, coeffs))
        torch.cuda.synchronize(); t3 = time.perf_counter()
        gen_tok = int(fit.sum())
        rows.append({"step": step, "warmup": step < a.warmup,
                     "t_inject": t1 - t0, "t_gen": t2 - t1, "t_update": t3 - t2,
                     "t_step": t3 - t0, "gen_tokens": gen_tok})

    timed = [r for r in rows if not r["warmup"]]
    def stat(k): v = [r[k] for r in timed]; return {"mean": float(np.mean(v)), "std": float(np.std(v))}
    peak_torch = torch.cuda.max_memory_allocated() / 1e9
    import subprocess
    smi = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                       "-i", os.environ["CUDA_VISIBLE_DEVICES"]], text=True).strip().splitlines()[0])
    out = {"method": "lora_es", "model": a.model, "N": N, "batch": B, "max_tokens": a.max_tokens,
           "seqs_per_step": N * B, "s_per_step": stat("t_step"), "t_inject": stat("t_inject"),
           "t_gen": stat("t_gen"), "t_update": stat("t_update"), "tokens_per_step": stat("gen_tokens"),
           "peak_vram_torch_gb": round(peak_torch, 2), "peak_vram_smi_mib": smi}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("RESULT:", json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
