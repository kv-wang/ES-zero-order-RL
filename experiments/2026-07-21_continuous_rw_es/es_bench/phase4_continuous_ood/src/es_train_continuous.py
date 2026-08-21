#!/usr/bin/env python
"""Phase 4 ES trainer with a continuous (lexicographic) reward option.

Mirrors es_bench/es_train.py EXACTLY for the binary arm (byte-identical Q1: same
data order, CRN batches, z-score shaping, fp32-base reconstruction, prefix cache
off). Adds --reward {binary,continuous}:
  binary     : fitness = primary = #correct/B                       (== Q1)
  continuous : fitness = primary + (MARGIN/B)*secondary             (lexicographic)
               secondary scored under member i's LIVE weights, right after its
               generate(), before the shift to member i+1 (scoring.score_secondary).

Per-step JSONL adds: primary[], secondary[], n_distinct_primary, pure_tiebreaker,
t_scoring, and keeps all Q1 telemetry. Checkpoints (fp16 base) saved at 0/50/100%.
Weight-drift ||theta-base|| tracked as a cheap KL-explosion proxy (full KL harness
runs at checkpoints in Phase 2).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=["binary", "continuous"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--population_size", type=int, required=True)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--pop_seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--ckpt_dir", default=None, help="if set, save fp16 base at 0/50/100%")
    p.add_argument("--eval_final", action="store_true",
                   help="run the OOD/ID battery in-process on the final trained weights")
    p.add_argument("--eval_cap", type=int, default=300)
    p.add_argument("--kl", action="store_true",
                   help="capture base-greedy NLL-drift KL proxy (step 0 vs final)")
    return p.parse_args()


def main():
    a = parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, HERE)
    import config as C
    import continuous_reward as cr
    import scoring

    model = a.model or C.MODEL
    num_steps = a.num_steps or C.NUM_STEPS
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench import data_math

    B, sigma, alpha, warmup = C.BATCH_SIZE, C.SIGMA, C.ALPHA, C.WARMUP
    train, _ = data_math.make_split(C.DATASET, C.TRAIN_SIZE, 200, C.DATA_SEED, levels=C.LEVELS)
    tok = AutoTokenizer.from_pretrained(model)

    # data order held constant across N — identical construction to Q1 es_train.py
    drng = random.Random(C.DATA_SEED)
    step_batches = [[drng.randrange(len(train)) for _ in range(B)] for _ in range(num_steps)]

    # max_model_len=4096 accommodates the in-process eval battery (hard MATH/olympiad
    # need long completions); greedy training generation (<=512 new tok) is unaffected.
    llm = LLM(model=model, dtype=C.DTYPE, gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              worker_extension_cls="es_bench.phase4_continuous_ood.src.phase4_worker.Phase4Worker")
    sp = SamplingParams(temperature=0.0, max_tokens=C.MAX_TOKENS, seed=42)
    llm.collective_rpc("es_snapshot_base")

    # KL proxy: capture base-greedy completions under the ORIGINAL base (before any update)
    kl_rec = None
    if a.kl:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
        import kl as klmod
        kl_prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
        kl_rec = klmod.capture_base(llm, kl_prompts, max_tokens=C.MAX_TOKENS)
        kl_base_self = klmod.drift(llm, tok, kl_rec)  # sanity: theta==base -> ~0
        print(f"[kl] captured {len(kl_rec)} base completions; base-vs-base drift={kl_base_self:.2e}", flush=True)

    prng = random.Random(a.pop_seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jf = open(a.out_prefix + ".jsonl", "w")
    ckpt_steps = {int(round(f * (num_steps - 1))): f for f in (0.0, 0.5, 1.0)} if a.ckpt_dir else {}
    n_zero = n_tiebreak = 0
    drift_sq = 0.0
    t_run0 = time.perf_counter()

    def save_ckpt(step, frac):
        d = Path(a.ckpt_dir) / f"step{step}_frac{int(frac*100)}"
        d.mkdir(parents=True, exist_ok=True)
        llm.collective_rpc("es_save_base", args=(str(d / "base_fp16.pt"),))

    for step in range(num_steps):
        if step in ckpt_steps:
            save_ckpt(step, ckpt_steps[step])
        batch = [train[i] for i in step_batches[step]]
        prompts = [data_math.build_prompt(tok, b["question"]) for b in batch]
        gts = [b["gt"] for b in batch]
        pop_seeds = [prng.randrange(2**31 - 1) for _ in range(a.population_size)]

        primary, secondary = [], []
        t_gen = t_swap = t_score = 0.0
        step_tokens = 0
        for seed in pop_seeds:
            t0 = time.perf_counter()
            llm.collective_rpc("es_set_member", args=(seed, sigma))
            t_swap += time.perf_counter() - t0

            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            t_gen += time.perf_counter() - t0
            gen_texts = [o.outputs[0].text for o in outs]

            rewards = [math_reward(t, gt)["reward"] for t, gt in zip(gen_texts, gts)]
            primary.append(float(np.mean(rewards)))
            step_tokens += int(sum(len(o.outputs[0].token_ids) for o in outs))

            if a.reward == "continuous":
                t0 = time.perf_counter()
                sec, _ = scoring.score_secondary(llm, tok, prompts, gen_texts, gts, C.ANSWER_OPENER)
                t_score += time.perf_counter() - t0
                secondary.append(sec)

        primary = np.array(primary, dtype=np.float64)
        if a.reward == "continuous":
            fitness = cr.combined_fitness(primary, secondary, B, C.TIEBREAK_MARGIN)
        else:
            fitness = primary.copy()

        coeffs, z = cr.zscore_coeffs(fitness, alpha, a.population_size)
        t0 = time.perf_counter()
        res = llm.collective_rpc("es_commit_update", args=(pop_seeds, coeffs.tolist()))
        t_update = time.perf_counter() - t0
        update_l2 = math.sqrt(max(float(res[0]), 0.0))
        drift_sq += float(res[0])

        zero_update = cr.is_zero_update(fitness)
        pure_tb = (a.reward == "continuous" and cr.is_pure_tiebreaker(primary, secondary))
        n_zero += int(zero_update); n_tiebreak += int(pure_tb)

        row = {
            "step": step, "warmup": step < warmup, "reward": a.reward,
            "population_size": a.population_size, "batch_size": B,
            "primary": primary.tolist(), "secondary": list(secondary),
            "fitness": fitness.tolist(), "fitness_std": float(fitness.std()),
            "n_distinct_primary": cr.n_distinct_primary(primary),
            "zero_update": zero_update, "pure_tiebreaker": pure_tb,
            "update_l2": update_l2, "drift_l2": math.sqrt(drift_sq),
            "tokens": step_tokens, "t_generation": t_gen, "t_perturb_swap": t_swap,
            "t_scoring": t_score, "t_update": t_update,
            "t_step": t_gen + t_swap + t_score + t_update,
        }
        jf.write(json.dumps(row) + "\n"); jf.flush()
        if step % 5 == 0 or step < warmup:
            print(f"[{a.reward} N={a.population_size} s{a.pop_seed}] step {step} "
                  f"fstd={row['fitness_std']:.4f} zero={zero_update} tb={pure_tb} "
                  f"|Δ|={update_l2:.2e} t_score={t_score:.2f}/{row['t_step']:.2f}s", flush=True)
    jf.close()

    # KL proxy on final theta (live weights == theta_final after the last commit)
    kl_drift = None
    if a.kl and kl_rec is not None:
        import kl as klmod
        kl_drift = klmod.drift(llm, tok, kl_rec)
        print(f"[kl] final base-greedy NLL drift D={kl_drift:.4f}", flush=True)

    rows = [json.loads(l) for l in open(a.out_prefix + ".jsonl")]
    timed = [r for r in rows if not r["warmup"]]
    def col(k): return [r[k] for r in timed]
    scoring_overhead = (np.sum(col("t_scoring")) / max(np.sum(col("t_step")), 1e-9)) if timed else 0.0
    summary = {
        "reward": a.reward, "model": model, "population_size": a.population_size,
        "pop_seed": a.pop_seed, "num_steps": num_steps,
        "zero_update_rate": n_zero / num_steps,
        "pure_tiebreaker_rate": n_tiebreak / num_steps,
        "final_drift_l2": rows[-1]["drift_l2"] if rows else 0.0,
        "kl_proxy_drift": kl_drift,
        "s_per_step_mean": float(np.mean(col("t_step"))) if timed else 0.0,
        "t_scoring_mean": float(np.mean(col("t_scoring"))) if timed else 0.0,
        "scoring_overhead_frac": float(scoring_overhead),
        "tokens_per_s": float(np.sum(col("tokens")) / max(np.sum(col("t_step")), 1e-9)) if timed else 0.0,
        "wall_clock_s": time.perf_counter() - t_run0,
    }
    # ---- in-process eval on the FINAL trained weights (live weights == theta_final) ----
    if a.eval_final:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
        import eval_core
        t0 = time.perf_counter()
        ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=2048,
                                   perq_prefix=a.out_prefix + "_finaleval")
        summary["eval_final"] = ev
        summary["eval_seconds"] = time.perf_counter() - t0
        print("EVAL_FINAL:", json.dumps(ev, indent=2), flush=True)

    with open(a.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
