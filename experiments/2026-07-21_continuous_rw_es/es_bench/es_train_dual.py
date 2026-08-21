#!/usr/bin/env python
"""Phase 3 dual-GPU vanilla ES: one process per GPU (CUDA_VISIBLE_DEVICES
isolation), population split evenly (N -> N/2 + N/2). The coordinator only
distributes member seeds and collects SCALAR fitness; each worker reconstructs
theta=base+sigma*eps for its members, evaluates, and returns fitness.

The ES update Sum_i coeff_i * eps_i is applied INDEPENDENTLY and IDENTICALLY on
both workers (all seeds+coeffs are shared, noise is deterministic per seed), so
the two base copies stay bit-identical with NO inter-GPU weight transfer. This
mirrors the single-GPU math exactly; only the population evaluation is parallel.

Reports dual-GPU s/step (barrier = slower worker's eval + update) and, with a
matching single-GPU es_train.py run, the 1->2 GPU scaling efficiency.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import multiprocessing as mp
from pathlib import Path
import numpy as np


def worker_proc(gpu, phys_gpu, args, in_q, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ["XDG_CONFIG_HOME"] = "/tmp/xdgconfig"; os.environ["VLLM_NO_USAGE_STATS"] = "1"
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench import data_math

    levels = [int(x) for x in args.levels.split(",") if x.strip()] or None
    train, _ = data_math.make_split(args.dataset, args.train_size, 200, args.data_seed, levels=levels)
    tok = AutoTokenizer.from_pretrained(args.model)
    drng = random.Random(args.data_seed)
    step_batches = [[drng.randrange(len(train)) for _ in range(args.batch_size)]
                    for _ in range(args.num_steps)]

    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False, worker_extension_cls="es_bench.es_worker.ESWorker")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42)
    llm.collective_rpc("es_snapshot_base")
    llm.collective_rpc("es_reset_peak_mem")
    out_q.put(("ready", gpu))

    while True:
        msg = in_q.get()
        if msg[0] == "stop":
            peak = llm.collective_rpc("es_max_mem")[0]
            out_q.put(("peak", gpu, peak)); break
        if msg[0] == "eval":
            _, step, my_seeds = msg
            batch = [train[i] for i in step_batches[step]]
            prompts = [data_math.build_prompt(tok, b["question"]) for b in batch]
            gts = [b["gt"] for b in batch]
            fit = {}; t_gen = t_swap = 0.0; toks = 0
            for seed in my_seeds:
                t0 = time.perf_counter(); llm.collective_rpc("es_set_member", args=(seed, args.sigma)); t_swap += time.perf_counter() - t0
                t0 = time.perf_counter(); outs = llm.generate(prompts, sp, use_tqdm=False); t_gen += time.perf_counter() - t0
                fit[seed] = float(np.mean([math_reward(o.outputs[0].text, gt)["reward"] for o, gt in zip(outs, gts)]))
                toks += int(sum(len(o.outputs[0].token_ids) for o in outs))
            out_q.put(("fit", gpu, fit, t_gen, t_swap, toks))
        elif msg[0] == "update":
            _, seeds, coeffs = msg
            t0 = time.perf_counter(); ds = llm.collective_rpc("es_commit_update", args=(seeds, coeffs))[0]
            out_q.put(("updated", gpu, time.perf_counter() - t0, float(ds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="math"); ap.add_argument("--levels", default="3,4,5")
    ap.add_argument("--population_size", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.001); ap.add_argument("--alpha", type=float, default=0.0005)
    ap.add_argument("--data_seed", type=int, default=1234); ap.add_argument("--pop_seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--max_tokens", type=int, default=512); ap.add_argument("--gpu_mem_util", type=float, default=0.5)
    ap.add_argument("--max_model_len", type=int, default=2048); ap.add_argument("--train_size", type=int, default=2000)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    assert len(gpus) == 2 and args.population_size % 2 == 0

    ctx = mp.get_context("spawn")
    qs = [(ctx.Queue(), ctx.Queue()) for _ in gpus]
    procs = [ctx.Process(target=worker_proc, args=(g, gpus[g], args, qs[g][0], qs[g][1])) for g in range(2)]
    for p in procs: p.start()

    def collect(kind, n=2, timeout=1800):
        """Collect exactly n messages of the given kind from both worker out-queues."""
        got = []
        t0 = time.perf_counter()
        while len(got) < n:
            for g in range(2):
                try:
                    m = qs[g][1].get_nowait()
                except Exception:
                    continue
                if m[0] == kind:
                    got.append(m)
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"timeout collecting {kind}: got {len(got)}/{n}")
            time.sleep(0.001)
        return got

    collect("ready")

    prng = random.Random(args.pop_seed)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jl = open(args.out_prefix + ".jsonl", "w")
    half = args.population_size // 2
    n_zero = 0; t_run0 = time.perf_counter()

    for step in range(args.num_steps):
        seeds = [prng.randrange(2**31 - 1) for _ in range(args.population_size)]
        split = {0: seeds[:half], 1: seeds[half:]}
        t0 = time.perf_counter()
        for g in range(2): qs[g][0].put(("eval", step, split[g]))
        fit = {}; per_worker = {}; toks = 0
        for m in collect("fit"):
            _, gg, f, tg, ts, tk = m
            fit.update(f); per_worker[gg] = (tg, ts); toks += tk
        t_eval = time.perf_counter() - t0  # barrier: both workers done

        f = np.array([fit[s] for s in seeds], dtype=np.float64)
        fmean, fstd, fvar = float(f.mean()), float(f.std()), float(f.var())
        z = (f - fmean) / (fstd + 1e-8)
        coeffs = ((args.alpha / args.population_size) * z).tolist()

        t0 = time.perf_counter()
        for g in range(2): qs[g][0].put(("update", seeds, coeffs))
        dsq = [m[3] for m in collect("updated")]
        t_update = time.perf_counter() - t0
        update_l2 = math.sqrt(max(dsq[0], 0.0)) if dsq else 0.0
        zero_update = (fstd < 1e-12) or (update_l2 < 1e-12); n_zero += int(zero_update)

        row = {"step": step, "warmup": step < args.warmup, "population_size": args.population_size,
               "batch_size": args.batch_size, "fitness": [fit[s] for s in seeds],
               "fitness_mean": fmean, "fitness_std": fstd, "fitness_var": fvar,
               "update_l2": update_l2, "zero_update": zero_update, "tokens": toks,
               "t_eval_barrier": t_eval, "t_update": t_update, "t_step": t_eval + t_update,
               "t_gen_gpu0": per_worker[0][0], "t_gen_gpu1": per_worker[1][0],
               "seqs": args.population_size * args.batch_size}
        jl.write(json.dumps(row) + "\n"); jl.flush()
        if step % 5 == 0:
            print(f"[dual N={args.population_size}] step {step} fmean={fmean:.3f} fstd={fstd:.3f} "
                  f"t_step={row['t_step']:.2f}s (eval {t_eval:.2f}+upd {t_update:.2f})", flush=True)

    for g in range(2): qs[g][0].put(("stop",))
    peaks = {}
    for m in collect("peak"):
        peaks[m[1]] = m[2]
    for p in procs: p.join(timeout=30)
    jl.close()

    rows = [json.loads(l) for l in open(args.out_prefix + ".jsonl")]
    timed = [r for r in rows if not r["warmup"]]
    def st(k): v = [r[k] for r in timed]; return {"mean": float(np.mean(v)), "std": float(np.std(v))}
    summary = {"mode": "dual_gpu", "model": args.model, "population_size": args.population_size,
               "batch_size": args.batch_size, "num_steps": args.num_steps, "warmup": args.warmup,
               "max_tokens": args.max_tokens, "gpus": gpus,
               "zero_update_rate": n_zero / args.num_steps,
               "s_per_step": st("t_step"), "t_eval_barrier": st("t_eval_barrier"),
               "t_update": st("t_update"), "seqs_per_step": args.population_size * args.batch_size,
               "tokens_per_step": st("tokens"),
               "peak_mem_torch_gb": {g: peaks[g]["max_allocated_bytes"] / 1e9 for g in peaks},
               "wall_clock_s": time.perf_counter() - t_run0}
    json.dump(summary, open(args.out_prefix + "_summary.json", "w"), indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
