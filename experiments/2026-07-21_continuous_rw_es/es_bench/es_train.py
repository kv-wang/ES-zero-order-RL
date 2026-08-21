#!/usr/bin/env python
"""Instrumented vanilla ES trainer (OpenAI-ES) for the Q1/Q2 benchmark.

- Vanilla ES: independent Gaussian perturbations, NO antithetic/mirrored pairs.
- z-score fitness shaping: z = (f - mean) / (std + 1e-8)  (repo default).
- Same reward object as GRPO: es_bench.shared_reward.math_reward.
- vLLM generation; prefix caching DISABLED so weight changes between members
  cannot leak KV/prefix state across members (protocol pitfall 6a).
- Resident fp32 base + reconstruct theta=base+sigma*eps per member; fp32 update
  accumulation on commit (protocol pitfall 6b). See es_worker.ESWorker.

Per-step JSONL row: fitness vector, fitness mean/std/var, update L2 norm,
zero_update flag, token count, and timing breakdown (generation / perturb+swap
/ update). Warmup steps are kept and flagged; timing summaries discard first 5.
"""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, threading, time
from pathlib import Path
from statistics import mean as _mean, pstdev

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--levels", default="", help="comma list, e.g. 3,4,5 (math only)")
    p.add_argument("--population_size", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8, help="tasks per member eval (B)")
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=0.0005)
    p.add_argument("--data_seed", type=int, default=1234, help="held constant across N")
    p.add_argument("--pop_seed", type=int, default=0, help="run seed (population noise)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


class SMISampler(threading.Thread):
    """Poll nvidia-smi for peak memory.used (MiB) on one GPU."""
    def __init__(self, gpu, period=0.5):
        super().__init__(daemon=True)
        self.gpu, self.period, self.peak, self._stop = gpu, period, 0, False
    def run(self):
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", str(self.gpu)], text=True).strip().splitlines()[0]
                self.peak = max(self.peak, int(out))
            except Exception:
                pass
            time.sleep(self.period)
    def stop(self):
        self._stop = True


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)   # physical GPU -> local cuda:0
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # in-process worker: RPCs + our sys.path
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from es_bench.shared_reward import math_reward
    from es_bench import data_math

    levels = [int(x) for x in args.levels.split(",") if x.strip()] or None
    train, _val = data_math.make_split(args.dataset, args.train_size, 200, args.data_seed, levels=levels)
    tok = AutoTokenizer.from_pretrained(args.model)

    # data order held constant across N: precompute per-step batch indices from data_seed
    drng = random.Random(args.data_seed)
    step_batches = [[drng.randrange(len(train)) for _ in range(args.batch_size)]
                    for _ in range(args.num_steps)]

    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              worker_extension_cls="es_bench.es_worker.ESWorker")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42)

    llm.collective_rpc("es_snapshot_base")
    llm.collective_rpc("es_reset_peak_mem")

    smi = SMISampler(args.gpu); smi.start()
    prng = random.Random(args.pop_seed)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jsonl = open(args.out_prefix + ".jsonl", "w")
    tol = 1e-12
    n_zero = 0
    t_run0 = time.perf_counter()

    for step in range(args.num_steps):
        batch = [train[i] for i in step_batches[step]]
        prompts = [data_math.build_prompt(tok, b["question"]) for b in batch]
        gts = [b["gt"] for b in batch]
        pop_seeds = [prng.randrange(2**31 - 1) for _ in range(args.population_size)]

        fitness, per_member_rewards = [], []
        t_gen = t_swap = 0.0
        step_tokens = 0
        for seed in pop_seeds:
            t0 = time.perf_counter()
            llm.collective_rpc("es_set_member", args=(seed, args.sigma))  # syncs internally
            t_swap += time.perf_counter() - t0

            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)              # blocking
            t_gen += time.perf_counter() - t0

            rewards = [math_reward(o.outputs[0].text, gt)["reward"] for o, gt in zip(outs, gts)]
            fitness.append(float(np.mean(rewards)))
            per_member_rewards.append(rewards)
            step_tokens += int(sum(len(o.outputs[0].token_ids) for o in outs))

        f = np.array(fitness, dtype=np.float64)
        fmean, fstd, fvar = float(f.mean()), float(f.std()), float(f.var())
        z = (f - fmean) / (fstd + 1e-8)
        coeffs = ((args.alpha / args.population_size) * z).tolist()

        t0 = time.perf_counter()
        res = llm.collective_rpc("es_commit_update", args=(pop_seeds, coeffs))
        t_update = time.perf_counter() - t0
        delta_sq = float(res[0])
        update_l2 = math.sqrt(max(delta_sq, 0.0))

        zero_update = (fstd < tol) or (update_l2 < tol)
        n_zero += int(zero_update)

        row = {
            "step": step, "warmup": step < args.warmup,
            "population_size": args.population_size, "batch_size": args.batch_size,
            "fitness": fitness, "fitness_mean": fmean, "fitness_std": fstd,
            "fitness_var": fvar, "update_l2": update_l2, "zero_update": zero_update,
            "tokens": step_tokens,
            "t_generation": t_gen, "t_perturb_swap": t_swap, "t_update": t_update,
            "t_step": t_gen + t_swap + t_update,
            "seqs": args.population_size * args.batch_size,
        }
        jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
        if step % 10 == 0 or step < args.warmup:
            print(f"[N={args.population_size} seed={args.pop_seed}] step {step} "
                  f"fmean={fmean:.3f} fstd={fstd:.3f} |Δ|={update_l2:.2e} "
                  f"zero={zero_update} t_step={row['t_step']:.2f}s", flush=True)

    jsonl.close()
    smi.stop(); time.sleep(0.6)
    peak = llm.collective_rpc("es_max_mem")[0]

    # timing summary over non-warmup steps
    rows = [json.loads(l) for l in open(args.out_prefix + ".jsonl")]
    timed = [r for r in rows if not r["warmup"]]
    def ms(key): return ([r[key] for r in timed])
    def stat(v): return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else {"mean": 0, "std": 0}

    summary = {
        "model": args.model, "dataset": args.dataset, "levels": levels,
        "population_size": args.population_size, "batch_size": args.batch_size,
        "sigma": args.sigma, "alpha": args.alpha,
        "data_seed": args.data_seed, "pop_seed": args.pop_seed, "gpu": args.gpu,
        "num_steps": args.num_steps, "warmup": args.warmup, "max_tokens": args.max_tokens,
        "zero_update_count": n_zero, "zero_update_rate": n_zero / args.num_steps,
        "fitness_var_mean": stat(ms("fitness_var"))["mean"],
        "s_per_step": stat(ms("t_step")),
        "t_generation": stat(ms("t_generation")),
        "t_perturb_swap": stat(ms("t_perturb_swap")),
        "t_update": stat(ms("t_update")),
        "seqs_per_step": args.population_size * args.batch_size,
        "tokens_per_step": stat(ms("tokens")),
        "tokens_per_s": float(np.sum(ms("tokens")) / max(np.sum(ms("t_step")), 1e-9)),
        "t_eff_per_effective_step": (stat(ms("t_step"))["mean"] / max(1 - n_zero / args.num_steps, 1e-9)),
        "peak_mem_torch_gb": peak["max_allocated_bytes"] / 1e9,
        "peak_mem_reserved_gb": peak["max_reserved_bytes"] / 1e9,
        "peak_mem_nvidia_smi_mib": smi.peak,
        "wall_clock_s": time.perf_counter() - t_run0,
    }
    with open(args.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
