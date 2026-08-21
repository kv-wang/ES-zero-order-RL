#!/usr/bin/env python
"""Base-axis probe trainer (full-weight). See base_axis_probe/PREREG.md.

--variant baseaxis : 4 anchor members (2 antithetic pairs along theta0-theta_t) + (N-4) Gaussian.
--variant vanilla  : N Gaussian members (matched control), identical otherwise.
Binary reward, z-score pool, mixed central-difference update. Per-step diagnostics logged.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseaxis", "vanilla"], required=True)
    p.add_argument("--population_size", type=int, required=True)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--pop_seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--a_max", type=float, default=0.1)
    p.add_argument("--anchor_threshold", type=float, default=1.0, help="min ||theta0-theta_t|| to activate anchors")
    p.add_argument("--kl", action="store_true")
    p.add_argument("--eval_final", action="store_true")
    p.add_argument("--eval_cap", type=int, default=300)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, HERE)
    import config as C

    N = a.population_size
    num_steps = a.num_steps or C.NUM_STEPS
    if a.variant == "baseaxis" and N <= 4:
        raise SystemExit("baseaxis needs N>4 (4 anchor members + >=1 Gaussian)")
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
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    drng = random.Random(C.DATA_SEED)
    step_batches = [[drng.randrange(len(train)) for _ in range(B)] for _ in range(num_steps)]

    llm = LLM(model=C.MODEL, dtype=C.DTYPE, gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              worker_extension_cls="es_bench.base_axis_probe.src.base_axis_worker.BaseAxisWorker")
    sp = SamplingParams(temperature=0.0, max_tokens=C.MAX_TOKENS, seed=42)
    llm.collective_rpc("es_snapshot_base")
    llm.collective_rpc("es_snapshot_theta0")
    d = llm.collective_rpc("es_param_count")[0]
    sigma_scale = sigma * math.sqrt(d)            # Gaussian perturbation norm ~ sigma*sqrt(d)

    kl_rec = None
    if a.kl:
        import kl as klmod
        kl_prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
        kl_rec = klmod.capture_base(llm, kl_prompts, max_tokens=C.MAX_TOKENS)
        print(f"[kl] base-vs-base drift={klmod.drift(llm, tok, kl_rec):.2e}", flush=True)

    prng = random.Random(a.pop_seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jf = open(a.out_prefix + ".jsonl", "w")
    cum_disp = 0.0
    t_run0 = time.perf_counter()

    def member_primary(prompts, gts):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        rewards = [math_reward(o.outputs[0].text, gt)["reward"] for o, gt in zip(outs, gts)]
        toks = int(sum(len(o.outputs[0].token_ids) for o in outs))
        return float(np.mean(rewards)), toks

    for step in range(num_steps):
        batch = [train[i] for i in step_batches[step]]
        prompts = [data_math.build_prompt(tok, b["question"]) for b in batch]
        gts = [b["gt"] for b in batch]

        axis_norm0 = llm.collective_rpc("es_axis_norm")[0]
        use_anchor = (a.variant == "baseaxis") and (axis_norm0 > a.anchor_threshold)

        anchors = []   # (a_k, sign) in fixed order: +a1,-a1,+a2,-a2
        if use_anchor:
            a1 = prng.uniform(0, a.a_max); a2 = prng.uniform(0, a.a_max)
            anchors = [(a1, +1), (a1, -1), (a2, +1), (a2, -1)]
        n_gauss = N - len(anchors)
        gauss_seeds = [prng.randrange(2**31 - 1) for _ in range(n_gauss)]

        fitness, step_tokens = [], 0
        for (av, sgn) in anchors:                       # anchor members first
            llm.collective_rpc("es_set_anchor_member", args=(av, sgn))
            f, tk = member_primary(prompts, gts); fitness.append(f); step_tokens += tk
        for seed in gauss_seeds:                         # then Gaussian members
            llm.collective_rpc("es_set_member", args=(seed, sigma))
            f, tk = member_primary(prompts, gts); fitness.append(f); step_tokens += tk

        f = np.array(fitness, dtype=np.float64)
        z = (f - f.mean()) / (f.std() + 1e-8)
        gauss_z = z[len(anchors):]
        gauss_coeffs = ((alpha / N) * gauss_z).tolist()

        s_anchor = 0.0; pair_fdiff = []
        if use_anchor:
            # pairs: (0:+a1,1:-a1) (2:+a2,3:-a2)
            for (pi, av) in [((0, 1), anchors[0][0]), ((2, 3), anchors[2][0])]:
                p_plus, p_minus = pi
                s_anchor += av * (z[p_plus] - z[p_minus])
                pair_fdiff.append(float(f[p_plus] - f[p_minus]))
            s_anchor *= (alpha / N) / sigma

        res = llm.collective_rpc("es_commit_update_baseaxis", args=(gauss_seeds, gauss_coeffs, s_anchor))[0]
        update_l2 = math.sqrt(max(res["delta_sq"], 0.0))
        signed_disp = res["signed_disp"]          # +toward base, -away
        cum_disp += signed_disp
        anchor_pert_norm = (np.mean([av for av, _ in anchors]) * axis_norm0) if anchors else 0.0
        scale_ratio = (sigma_scale / anchor_pert_norm) if anchor_pert_norm > 0 else float("inf")

        row = {
            "step": step, "warmup": step < warmup, "variant": a.variant,
            "population_size": N, "use_anchor": use_anchor,
            "axis_norm": axis_norm0, "signed_disp": signed_disp, "cum_disp": cum_disp,
            "pair_fdiff": pair_fdiff, "s_anchor": s_anchor,
            "sigma_scale": sigma_scale, "anchor_pert_norm": anchor_pert_norm,
            "scale_ratio": scale_ratio, "scale_flag": bool(scale_ratio > 10 or scale_ratio < 0.1),
            "fitness_std": float(f.std()), "zero_update": bool(f.std() < 1e-12),
            "update_l2": update_l2, "tokens": step_tokens,
        }
        jf.write(json.dumps(row) + "\n"); jf.flush()
        if step % 5 == 0 or step < warmup:
            fd = ",".join(f"{x:+.3f}" for x in pair_fdiff) if pair_fdiff else "-"
            print(f"[{a.variant} N={N} s{a.pop_seed}] step {step} axis={axis_norm0:.1f} "
                  f"disp={signed_disp:+.3f} cum={cum_disp:+.2f} f+-f-=[{fd}] "
                  f"anchor={use_anchor} |Δ|={update_l2:.2e}", flush=True)
    jf.close()

    kl_drift = None
    if a.kl and kl_rec is not None:
        import kl as klmod
        kl_drift = klmod.drift(llm, tok, kl_rec)
        print(f"[kl] final drift D={kl_drift:.4f}", flush=True)

    rows = [json.loads(l) for l in open(a.out_prefix + ".jsonl")]
    timed = [r for r in rows if not r["warmup"]]
    summary = {
        "variant": a.variant, "model": C.MODEL, "population_size": N, "pop_seed": a.pop_seed,
        "num_steps": num_steps, "a_max": a.a_max, "anchor_threshold": a.anchor_threshold,
        "final_axis_norm": rows[-1]["axis_norm"] if rows else 0.0,
        "final_cum_disp": cum_disp,
        "mean_signed_disp": float(np.mean([r["signed_disp"] for r in timed])) if timed else 0.0,
        "mean_pair_fdiff": float(np.mean([x for r in timed for x in r["pair_fdiff"]])) if any(r["pair_fdiff"] for r in timed) else None,
        "steps_scale_flagged": int(sum(r["scale_flag"] for r in timed)),
        "kl_proxy_drift": kl_drift,
        "s_per_step_mean": (time.perf_counter() - t_run0) / num_steps,
        "wall_clock_s": time.perf_counter() - t_run0,
    }
    if a.eval_final:
        import eval_core
        t0 = time.perf_counter()
        summary["eval_final"] = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=2048,
                                                      perq_prefix=a.out_prefix + "_finaleval")
        summary["eval_seconds"] = time.perf_counter() - t0
    with open(a.out_prefix + "_summary.json", "w") as f2:
        json.dump(summary, f2, indent=2)
    print("SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "eval_final"}, indent=2), flush=True)
    if "eval_final" in summary:
        print("EVAL_FINAL:", json.dumps(summary["eval_final"], indent=2), flush=True)


if __name__ == "__main__":
    main()
