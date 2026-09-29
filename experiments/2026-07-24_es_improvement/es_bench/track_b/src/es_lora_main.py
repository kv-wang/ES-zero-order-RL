#!/usr/bin/env python
"""Phase B -- LoRA-ES main arm at large B (overnight run).

N members = N LoRA adapters over one frozen base, ONE batched multi-adapter generation per step
(chunked to fit KV), CRN: all members see the SAME B prompts, resampled every step.

Telemetry per step (JSONL): per-member fitness, per-member per-problem correctness bitmap (hex),
z-scores, number of distinct fitness levels, split-half Spearman of the member ranking, update
norm, wall-clock split into generation vs update, tokens generated, truncation rate, chunk count.
Probe-KL to base every --probe_every steps on a fixed 32-prompt set. Adapter checkpoints every
--ckpt_every steps.

Split-half Spearman is the signal detector: the B problems are split into two random halves,
each member scored on each half, and the two rank orderings correlated. If the member ranking is
shot noise, rho ~ 0 and the ES update direction is meaningless no matter how large alpha is.

Auto-gates (exit codes let the driver relaunch once, unattended):
  90  step-`gate_step` check failed (train-fitness slope <= 0 AND median split-half rho < 0.1)
  91  KL guard tripped (probe KL > gate_kl before step 50)
  0   completed, or stopped cleanly at --deadline_ts
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["gsm8k", "math", "countdown"], required=True)
    p.add_argument("--population_size", type=int, default=30)
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--pop_seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=None,
                   help="LoRA rank override (default lora_cfg.LORA_RANK=16)")
    p.add_argument("--targets", default=None,
                   help="comma list of target modules (default lora_cfg.TARGET_MODULES=attn+mlp)")
    p.add_argument("--antithetic", action="store_true",
                   help="N/2 mirrored pairs (+eps,-eps); N must be even. Central-difference "
                        "estimator at identical generation cost (WALLCLOCK_ANALYSIS sec.5)")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--chunk", type=int, default=2000, help="max sequences per generate() call")
    p.add_argument("--ckpt_every", type=int, default=25)
    p.add_argument("--probe_every", type=int, default=25)
    p.add_argument("--kl_probe_n", type=int, default=32)
    p.add_argument("--gate_step", type=int, default=60)
    p.add_argument("--gate_rho", type=float, default=0.1)
    p.add_argument("--gate_kl", type=float, default=3e-2)
    p.add_argument("--no_gates", action="store_true")
    p.add_argument("--deadline_ts", type=float, default=0.0, help="unix ts; stop cleanly after")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--fresh", action="store_true",
                   help="ignore an existing <out_prefix>.jsonl and start over (default is to "
                        "resume: replay the logged updates exactly, then continue)")
    return p.parse_args()


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return float(num / (da * db)) if da > 0 and db > 0 else 0.0


def slope(y):
    n = len(y)
    if n < 3:
        return 0.0
    x = list(range(n))
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx) if sxx else 0.0


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
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "track_a", "src"))

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from es_bench.shared_reward import math_reward
    from es_bench import data_math
    import lora_cfg
    import config as C

    N, B = a.population_size, a.batch
    if a.antithetic and N % 2 != 0:
        sys.exit("--antithetic requires even population_size")
    rank = a.rank if a.rank is not None else lora_cfg.LORA_RANK
    targets = a.targets.split(",") if a.targets else lora_cfg.TARGET_MODULES
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.dataset == "countdown":
        import data_countdown as dc
        train, val = dc.load_split()
        mk_prompt = lambda r: r["context"]                       # raw completion, no template
        row_correct = lambda text, r: int(dc.reward(text, r) >= 1.0)
    else:
        levels = [3, 4, 5] if a.dataset == "math" else None
        train, val = data_math.make_split(a.dataset, 2000, 300, a.data_seed, levels=levels)
        mk_prompt = lambda r: data_math.build_prompt(tok, r["question"])
        row_correct = lambda text, r: int(math_reward(text, r["gt"])["reward"] >= 1.0)
    tmpl = lora_cfg.build_template(a.model, rank=rank, targets=targets)
    d_adapter = sum(t["shapeA"][0] * t["shapeA"][1] + t["shapeB"][0] * t["shapeB"][1]
                    for t in tmpl)

    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir or (a.out_prefix + "_ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # max_model_len must cover prompt + generation: 2048 was fine for max_tokens<=512 but
    # silently starves the cap-2048 protocol (MATH prompts run up to ~400 tokens).
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=C.TRAIN_GPU_MEM_UTIL,
              max_model_len=a.max_tokens + 1024,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
              enable_lora=True, max_loras=N, max_lora_rank=rank,
              max_cpu_loras=2 * N + 8,
              worker_extension_cls="es_bench.track_b.src.lora_main_worker.LoRAMainWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, rank, lora_cfg.LORA_ALPHA, targets, 0))
    print(f"[cfg] rank={rank} targets={targets} d={d_adapter:,} antithetic={a.antithetic} "
          f"util={C.TRAIN_GPU_MEM_UTIL}", flush=True)
    THETA_ID = 990000

    # ---- fixed KL probe set, captured under the BASE model (no adapter) ----
    klp = [mk_prompt(val[300 - a.kl_probe_n + i]) for i in range(a.kl_probe_n)]
    spb = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=0)
    kl_base = []
    for p, o in zip(klp, llm.generate(klp, spb, use_tqdm=False)):
        c = o.outputs[0]
        pid = tok(p, add_special_tokens=False)["input_ids"]
        kl_base.append({"ids": pid + list(c.token_ids), "a": len(pid),
                        "b": len(pid) + len(c.token_ids),
                        "lp": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)]})

    def probe_kl(step):
        tid = THETA_ID + step
        llm.collective_rpc("es_lora_inject_theta", args=(tid,))
        req = LoRARequest(f"theta{step}", tid, "/es/inmem")
        sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        reqs = [TokensPrompt(prompt_token_ids=r["ids"]) for r in kl_base]
        outs = llm.generate(reqs, sp2, lora_request=[req] * len(kl_base), use_tqdm=False)
        num = den = 0.0
        for r, out in zip(kl_base, outs):
            plp = out.prompt_logprobs
            for pos in range(r["a"], r["b"]):
                e = plp[pos] if pos < len(plp) else None
                if e and r["ids"][pos] in e:
                    num += (r["lp"][pos - r["a"]] - e[r["ids"][pos]].logprob)
                    den += 1
        return float(num / den) if den else 0.0

    def save_ckpt(step, tag=""):
        st = llm.collective_rpc("es_lora_get_theta")[0]
        p = ckpt_dir / f"theta_step{step}{tag}.pt"
        torch.save(st, p)
        return str(p)

    prng = np.random.RandomState(a.pop_seed)
    drng = random.Random(a.data_seed)
    hrng = random.Random(a.data_seed + 777)
    rows_meta, fit_hist, rho_hist, kl_curve, ckpts = [], [], [], [], []
    stop_reason = "completed"

    # ---- resume: replay logged updates exactly (theta is deterministic from init + seeds+z) ----
    jpath = a.out_prefix + ".jsonl"
    resume_rows = []
    if not a.fresh and os.path.exists(jpath) and os.path.getsize(jpath) > 0:
        resume_rows = [json.loads(l) for l in open(jpath) if l.strip()]
    R = len(resume_rows)
    if R:
        print(f"[resume] {R} completed steps in {jpath}; replaying updates on theta", flush=True)
        for r_i, row in enumerate(resume_rows):
            if bool(row.get("antithetic")) != bool(a.antithetic):
                sys.exit(f"[resume] antithetic flag mismatch at logged step {r_i}")
            if a.antithetic:
                pair_seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(N // 2)]
                exp_seeds = [pair_seeds[i // 2] for i in range(N)]
            else:
                exp_seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(N)]
            if exp_seeds != row["seeds"]:
                sys.exit(f"[resume] pop_seed stream mismatch at logged step {r_i} -- "
                         "config differs from the original run; use --fresh to start over")
            for _ in range(B):                      # fast-forward the batch-index stream
                drng.randrange(len(train))
            half = list(range(B))                    # fast-forward the split-half stream
            hrng.shuffle(half)
            coeffs = [(a.alpha / N) * zi for zi in row["z"]]
            if a.antithetic:
                pair_coeffs = [coeffs[2 * p] - coeffs[2 * p + 1] for p in range(N // 2)]
                upd_sq = llm.collective_rpc("es_lora_commit",
                                            args=(row["seeds"][::2], pair_coeffs))[0]
            else:
                upd_sq = llm.collective_rpc("es_lora_commit", args=(row["seeds"], coeffs))[0]
            got = math.sqrt(max(upd_sq, 0.0))
            if row["update_l2"] > 1e-9 and abs(got - row["update_l2"]) > 1e-5 * row["update_l2"]:
                sys.exit(f"[resume] replay diverged at step {r_i}: |D| {got:.6f} vs "
                         f"logged {row['update_l2']:.6f}")
        fit_hist = [r["fit_mean"] for r in resume_rows]
        rho_hist = [r["split_half_spearman"] for r in resume_rows]
        kR = probe_kl(R)
        kl_curve.append({"step": R, "kl": kR, "resumed": True})
        ckpts.append({"step": R, "path": save_ckpt(R, "_resume")})
        print(f"[resume] replay verified ({R} steps, all |D| match); KL@{R}={kR:.5f}; "
              f"continuing at step {R}", flush=True)
        jf = open(jpath, "a")
    else:
        jf = open(jpath, "w")
        kl0 = probe_kl(0)
        kl_curve.append({"step": 0, "kl": kl0})
        print(f"[main {a.dataset} N{N} B{B} sig{a.sigma} a{a.alpha}] step0 KL={kl0:.5f}",
              flush=True)
        ckpts.append({"step": 0, "path": save_ckpt(0)})

    a.resumed_from_step = R
    t_run0 = time.perf_counter()
    for step in range(R, a.num_steps):
        t_s0 = time.perf_counter()
        idx = [drng.randrange(len(train)) for _ in range(B)]
        batch = [train[i] for i in idx]
        prompts = [mk_prompt(b) for b in batch]
        if a.antithetic:
            # N/2 unique eps, each realized as +eps and -eps. NOTE the pop_seed stream draws
            # N/2 ints/step here vs N in the vanilla arm -- replay tooling must branch on the
            # 'antithetic' flag in the summary/rows.
            pair_seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(N // 2)]
            seeds = [pair_seeds[i // 2] for i in range(N)]
            signs = [1.0 if i % 2 == 0 else -1.0 for i in range(N)]
        else:
            pair_seeds, signs = None, None
            seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(N)]
        ids = llm.collective_rpc("es_lora_inject_members",
                                 args=(1 + step * N, seeds, a.sigma, signs))[0]

        reqs, sps, lrs = [], [], []
        for i, lid in enumerate(ids):
            lr = LoRARequest(f"s{step}m{i}", lid, "/es/inmem")
            for b in range(B):
                reqs.append(prompts[b])
                lrs.append(lr)
                sps.append(SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=seeds[i]))

        t_g0 = time.perf_counter()
        outs, n_chunks = [], 0
        for lo in range(0, len(reqs), a.chunk):
            hi = min(lo + a.chunk, len(reqs))
            outs.extend(llm.generate(reqs[lo:hi], sps[lo:hi], lora_request=lrs[lo:hi],
                                     use_tqdm=False))
            n_chunks += 1
        t_gen = time.perf_counter() - t_g0

        correct = np.zeros((N, B), dtype=np.int8)
        n_tok = n_trunc = 0
        for i in range(N):
            for b in range(B):
                o = outs[i * B + b].outputs[0]
                correct[i, b] = row_correct(o.text, batch[b])
                n_tok += len(o.token_ids)
                n_trunc += int(o.finish_reason == "length")
        fit = correct.mean(axis=1).astype(np.float64)

        half = list(range(B))
        hrng.shuffle(half)
        h1, h2 = half[: B // 2], half[B // 2:]
        f1 = correct[:, h1].mean(axis=1).tolist()
        f2 = correct[:, h2].mean(axis=1).tolist()
        rho = spearman(f1, f2)

        z = (fit - fit.mean()) / (fit.std() + 1e-8)
        coeffs = ((a.alpha / N) * z).tolist()
        t_u0 = time.perf_counter()
        if a.antithetic:
            # Delta = (alpha/N) * sum_p (z+_p - z-_p) * eps_p  -- fold the mirrored member's sign
            # into a per-pair coefficient so es_lora_commit (which knows nothing of signs) sums
            # over the N/2 unique seeds only.
            pair_coeffs = [coeffs[2 * p] - coeffs[2 * p + 1] for p in range(N // 2)]
            upd_sq = llm.collective_rpc("es_lora_commit", args=(pair_seeds, pair_coeffs))[0]
        else:
            upd_sq = llm.collective_rpc("es_lora_commit", args=(seeds, coeffs))[0]
        t_upd = time.perf_counter() - t_u0

        fit_hist.append(float(fit.mean()))
        rho_hist.append(rho)
        row = {
            "step": step, "seeds": seeds, "antithetic": a.antithetic,
            "fit_mean": float(fit.mean()), "fit_std": float(fit.std()),
            "fitness": fit.tolist(), "z": z.tolist(),
            "n_distinct_fitness": int(len(set(fit.tolist()))),
            "split_half_spearman": rho,
            "zero_update": bool(fit.std() < 1e-12),
            "update_l2": math.sqrt(max(upd_sq, 0.0)),
            "t_step": time.perf_counter() - t_s0, "t_generation": t_gen, "t_update": t_upd,
            "tokens": n_tok, "trunc_rate": n_trunc / float(N * B), "n_chunks": n_chunks,
            "correct_bitmap_hex": ["%x" % int("".join(map(str, correct[i].tolist())), 2)
                                   for i in range(N)],
        }
        jf.write(json.dumps(row) + "\n")
        jf.flush()

        if (step + 1) % a.probe_every == 0 or step == 0:
            k = probe_kl(step + 1)
            kl_curve.append({"step": step + 1, "kl": k})
            print(f"[main {a.dataset}] step {step+1} fit={fit.mean():.4f} rho={rho:+.3f} "
                  f"KL={k:.5f} |D|={row['update_l2']:.3f} {row['t_step']:.1f}s/step "
                  f"chunks={n_chunks} trunc={row['trunc_rate']:.3f}", flush=True)
            if not a.no_gates and step + 1 < 50 and k > a.gate_kl:
                stop_reason = f"KL_GUARD kl={k:.4g}>{a.gate_kl:g} at step {step+1}"
                print(f"[GATE] {stop_reason}", flush=True)
                jf.close()
                _write_summary(a, N, B, fit_hist, rho_hist, kl_curve, ckpts, stop_reason,
                               time.perf_counter() - t_run0, rows_meta)
                sys.exit(91)

        if (step + 1) % a.ckpt_every == 0:
            ckpts.append({"step": step + 1, "path": save_ckpt(step + 1)})

        if not a.no_gates and (step + 1) == a.gate_step:
            sl = slope(fit_hist)
            med_rho = float(np.median(rho_hist))
            ok = not (sl <= 0 and med_rho < a.gate_rho)
            print(f"[GATE] step-{a.gate_step} check: fit_slope={sl:+.3e} median_rho={med_rho:+.3f} "
                  f"-> {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                stop_reason = (f"STEP{a.gate_step}_GATE slope={sl:.3e}<=0 and "
                               f"median_rho={med_rho:.3f}<{a.gate_rho}")
                jf.close()
                _write_summary(a, N, B, fit_hist, rho_hist, kl_curve, ckpts, stop_reason,
                               time.perf_counter() - t_run0, rows_meta)
                sys.exit(90)

        if a.deadline_ts and time.time() > a.deadline_ts:
            stop_reason = f"DEADLINE reached at step {step+1}/{a.num_steps}"
            print(f"[stop] {stop_reason}", flush=True)
            break

    jf.close()
    ckpts.append({"step": "final", "path": save_ckpt(len(fit_hist), "_final")})
    _write_summary(a, N, B, fit_hist, rho_hist, kl_curve, ckpts, stop_reason,
                   time.perf_counter() - t_run0, rows_meta)
    print("[done]", stop_reason, flush=True)


def _write_summary(a, N, B, fit_hist, rho_hist, kl_curve, ckpts, stop_reason, wall, rows_meta):
    best_i = int(np.argmax(fit_hist)) if fit_hist else -1
    s = {
        "model": a.model, "dataset": a.dataset, "population_size": N, "batch": B,
        "rank": a.rank, "targets": a.targets, "antithetic": a.antithetic,
        "sigma": a.sigma, "alpha": a.alpha, "num_steps_requested": a.num_steps,
        "steps_completed": len(fit_hist), "pop_seed": a.pop_seed, "data_seed": a.data_seed,
        "max_tokens": a.max_tokens, "chunk": a.chunk, "stop_reason": stop_reason,
        "resumed_from_step": getattr(a, "resumed_from_step", 0),
        "wall_clock_s": wall,
        # wall covers only the steps run in THIS process (resume replays are ~free)
        "s_per_step_mean": (wall / (len(fit_hist) - getattr(a, "resumed_from_step", 0)))
                           if len(fit_hist) > getattr(a, "resumed_from_step", 0) else None,
        "fit_first20": float(np.mean(fit_hist[:20])) if fit_hist else None,
        "fit_last20": float(np.mean(fit_hist[-20:])) if fit_hist else None,
        "fit_slope_per_step": slope(fit_hist),
        "median_split_half_spearman": float(np.median(rho_hist)) if rho_hist else None,
        "mean_split_half_spearman": float(np.mean(rho_hist)) if rho_hist else None,
        "best_fit_step": best_i, "best_fit": (fit_hist[best_i] if best_i >= 0 else None),
        "kl_curve": kl_curve, "checkpoints": ckpts,
    }
    json.dump(s, open(a.out_prefix + "_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
