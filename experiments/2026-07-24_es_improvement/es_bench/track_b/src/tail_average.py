#!/usr/bin/env python
"""Phase 1 -- tail averaging vs pure shrinkage, on an already-trained ES run.

Premise (Phase 0): the ES update has constant norm and near-random direction -- a
drunkard's walk. If a coherent drift is buried in it, averaging the tail of the
trajectory should cancel the lateral oscillation and expose the net displacement.

The confound this script is built around: averaging checkpoints also SHRINKS the
iterate back toward theta_0, and shrinkage alone lowers KL and recovers accuracy
without extracting any direction. So every averaging arm is compared against a
pure-shrinkage control on the same trajectory,

    theta_lambda = theta_0 + lambda * (theta_final - theta_0),

which encodes "no signal extracted, just step back a bit". Averaging only counts as
having found a direction if it beats the shrinkage family AT MATCHED KL.

No retraining and no token generation: theta_t is reconstructed exactly from the
logged fitness vectors and the seed stream (see tailavg_worker.TailAvgWorker).

Usage:
  python tail_average.py --jsonl results/exp3b_math/fulles_N30.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct --alpha 5e-4 --pop_seed 0 \
      --tail 160,170,180,190,200 --tail_dense 50 --lambdas 0.25,0.5,0.75 \
      --eval_cap 300 --gpu 0 --out results/phase1_tailavg/fulles_N30
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="per-step log of the trained run")
    p.add_argument("--model", required=True)
    p.add_argument("--alpha", type=float, required=True, help="must match the training run")
    p.add_argument("--pop_seed", type=int, default=0)
    p.add_argument("--tail", default="160,170,180,190,200",
                   help="sparse checkpoint set T (theta_t = after t updates)")
    p.add_argument("--tail_dense", type=int, default=50,
                   help="also average the last K consecutive checkpoints (0=skip)")
    p.add_argument("--lambdas", default="0.25,0.5,0.75", help="pure-shrinkage control grid")
    p.add_argument("--eval_cap", type=int, default=300)
    p.add_argument("--eval_max_tokens", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=20, help="steps per replay RPC")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--skip_eval", action="store_true", help="replay + validate only")
    return p.parse_args()


def tail_coeffs(T, num_steps):
    """c_s = |{t in T : s < t}| / |T|, so theta_0 + sum_s c_s Delta_s = mean_{t in T} theta_t."""
    k = len(T)
    return [sum(1 for t in T if s < t) / k for s in range(num_steps)]


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, HERE)

    import config as C
    import continuous_reward as cr
    import eval_core
    import kl as klmod
    from transformers import AutoTokenizer
    from vllm import LLM

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    # ---------- host-side trajectory reconstruction ----------
    rows = [json.loads(l) for l in open(a.jsonl)]
    N = int(rows[0]["population_size"])
    num_steps = len(rows)
    T = sorted(int(x) for x in a.tail.split(",") if x.strip())
    if max(T) > num_steps:
        raise SystemExit(f"tail checkpoint {max(T)} exceeds num_steps={num_steps}")
    lambdas = [float(x) for x in a.lambdas.split(",") if x.strip()]

    prng = random.Random(a.pop_seed)
    seed_stream, coeff_stream = [], []
    for r in rows:
        seed_stream.append([prng.randrange(2**31 - 1) for _ in range(N)])
        coeffs, _ = cr.zscore_coeffs(r["fitness"], a.alpha, N)
        coeff_stream.append(coeffs.tolist())

    # accumulator 0 = theta_final - theta_0 ; 1 = sparse tail avg ; 2 = dense tail avg
    acc_names = ["final", f"avg{len(T)}"]
    weights_per_step = [[1.0] for _ in range(num_steps)]
    c_sparse = tail_coeffs(T, num_steps)
    for s in range(num_steps):
        weights_per_step[s].append(c_sparse[s])
    T_dense = None
    if a.tail_dense:
        T_dense = list(range(num_steps - a.tail_dense + 1, num_steps + 1))
        c_dense = tail_coeffs(T_dense, num_steps)
        acc_names.append(f"avg{len(T_dense)}dense")
        for s in range(num_steps):
            weights_per_step[s].append(c_dense[s])
    n_acc = len(acc_names)

    print(f"[cfg] {a.jsonl}: N={N} steps={num_steps} alpha={a.alpha} pop_seed={a.pop_seed}", flush=True)
    print(f"[cfg] tail T={T}  dense={'last %d' % a.tail_dense if T_dense else 'off'}  "
          f"lambdas={lambdas}  accumulators={acc_names}", flush=True)

    # ---------- engine ----------
    # gpu_memory_utilization is lowered vs training (0.85): the accumulators live
    # OUTSIDE the vLLM budget (n_acc+1 fp32 copies of the model). Greedy decoding makes
    # eval insensitive to KV size; theta_final is re-evaluated below as a cross-check.
    llm = LLM(model=a.model, dtype=C.DTYPE, gpu_memory_utilization=0.40,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              worker_extension_cls="es_bench.track_b.src.tailavg_worker.TailAvgWorker")
    tok = AutoTokenizer.from_pretrained(a.model)
    llm.collective_rpc("es_snapshot_base")

    # KL record captured under the ORIGINAL base, same prompt set as training
    kl_prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
    kl_rec = klmod.capture_base(llm, kl_prompts, max_tokens=C.MAX_TOKENS)
    kl_self = klmod.drift(llm, tok, kl_rec)
    print(f"[kl] captured {len(kl_rec)} base completions; base-vs-base drift={kl_self:.3e}", flush=True)

    # ---------- replay ----------
    llm.collective_rpc("es_accum_init", args=(n_acc,))
    t0 = time.perf_counter()
    replay_l2, worst = [], 0.0
    for lo in range(0, num_steps, a.chunk):
        hi = min(lo + a.chunk, num_steps)
        chunk = [(seed_stream[s], coeff_stream[s], weights_per_step[s]) for s in range(lo, hi)]
        dsq = llm.collective_rpc("es_replay_chunk", args=(chunk,))[0]
        for s, d in zip(range(lo, hi), dsq):
            got = math.sqrt(max(d, 0.0))
            want = float(rows[s]["update_l2"])
            replay_l2.append(got)
            denom = max(want, 1e-12)
            worst = max(worst, abs(got - want) / denom)
        print(f"[replay] steps {lo}-{hi-1}  worst rel.err so far={worst:.2e}", flush=True)
    replay_s = time.perf_counter() - t0
    print(f"[replay] done in {replay_s:.1f}s; max |reconstructed-logged|/logged over "
          f"{num_steps} steps = {worst:.3e}", flush=True)

    VALID_TOL = 1e-4
    if worst > VALID_TOL:
        print(f"[replay] FATAL: reconstruction does not match the logged update norms "
              f"(tol {VALID_TOL:.0e}). Refusing to report evals from a wrong trajectory.", flush=True)
        json.dump({"error": "replay_validation_failed", "max_rel_err": worst,
                   "jsonl": a.jsonl}, open(a.out + "_summary.json", "w"), indent=2)
        return

    # ---------- variants ----------
    def w_final(x):       # scale accumulator 0 -> shrinkage family
        return [x] + [0.0] * (n_acc - 1)

    def unit(j):
        w = [0.0] * n_acc
        w[j] = 1.0
        return w

    # Norm probe FIRST. Tail averaging turns out to shrink only slightly (it can cancel
    # oscillation inside the window but not the walk accumulated before it), so a fixed
    # lambda grid can miss the averaging arm's scale entirely and leave the two families
    # non-overlapping in KL -- i.e. no matched comparison at all. Measure each average's
    # effective shrinkage lambda_eff = ||A_j|| / ||theta_final - theta_0|| and add a
    # shrinkage control at exactly that norm, so "averaged" vs "just stepped back" are
    # compared at matched displacement rather than by interpolation.
    norms = [llm.collective_rpc("es_set_theta", args=(unit(j),))[0] for j in range(n_acc)]
    lam_eff = [norms[j] / norms[0] if norms[0] > 0 else float("nan") for j in range(n_acc)]
    for j in range(1, n_acc):
        print(f"[norm] {acc_names[j]}: ||A||={norms[j]:.2f} vs ||final||={norms[0]:.2f} "
              f"-> lambda_eff={lam_eff[j]:.4f}", flush=True)

    variants = [("theta_final", w_final(1.0), "trajectory endpoint (lambda=1)")]
    for j in range(1, n_acc):
        variants.append((f"tailavg_{acc_names[j]}", unit(j), "tail average"))
    for lam in lambdas:
        variants.append((f"shrink_lam{lam:g}", w_final(lam), "pure-shrinkage control"))
    for j in range(1, n_acc):
        lam = lam_eff[j]
        if not (lam == lam) or any(abs(lam - x) < 1e-3 for x in lambdas):
            continue
        variants.append((f"shrink_matched_{acc_names[j]}", w_final(lam),
                         f"pure-shrinkage control at matched norm (lambda={lam:.4f}) for {acc_names[j]}"))

    results = {}
    for tag, w, kind in variants:
        t1 = time.perf_counter()
        drift = llm.collective_rpc("es_set_theta", args=(w,))[0]
        kl = klmod.drift(llm, tok, kl_rec)
        ev = None
        if not a.skip_eval:
            ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=a.eval_max_tokens,
                                       perq_prefix=f"{a.out}_{tag}")
        results[tag] = {"kind": kind, "weights": w, "drift_l2": drift, "kl": kl, "eval": ev,
                        "seconds": time.perf_counter() - t1}
        m500 = (ev or {}).get("math500", {})
        print(f"[variant] {tag:24s} drift={drift:8.2f} KL={kl:9.4e} "
              f"L3-5={m500.get('primary_L3_5_accuracy')} ({time.perf_counter()-t1:.0f}s)", flush=True)

    # base sanity: theta_0 itself
    drift0 = llm.collective_rpc("es_set_theta", args=([0.0] * n_acc,))[0]
    kl0 = klmod.drift(llm, tok, kl_rec)
    print(f"[sanity] theta_0 restored: drift={drift0:.3e} KL={kl0:.3e} (both should be ~0)", flush=True)

    summary = {
        "jsonl": a.jsonl, "model": a.model, "N": N, "num_steps": num_steps,
        "alpha": a.alpha, "pop_seed": a.pop_seed,
        "tail_sparse": T, "tail_dense": T_dense, "lambdas": lambdas,
        "accumulators": acc_names, "eval_cap": a.eval_cap,
        "accum_norms": dict(zip(acc_names, norms)),
        "lambda_eff": {acc_names[j]: lam_eff[j] for j in range(1, n_acc)},
        "replay": {"seconds": replay_s, "max_rel_err_vs_logged_update_l2": worst,
                   "validated": True, "tol": VALID_TOL},
        "kl_base_self": kl_self,
        "sanity_theta0": {"drift_l2": drift0, "kl": kl0},
        "variants": results,
    }
    json.dump(summary, open(a.out + "_summary.json", "w"), indent=2)
    print(f"[done] wrote {a.out}_summary.json", flush=True)
    llm.collective_rpc("es_accum_free")


if __name__ == "__main__":
    main()
