#!/usr/bin/env python
"""LoRA-ES trainer (A1 sanity + sweeps). N members = N LoRA adapters over one frozen base;
ONE batched generate per step (N*B requests), real reward, z-score OpenAI-ES update on adapter
params only. Logs per-step fitness/update; periodic ID accuracy (under the mean adapter Theta)
and KL-to-base proxy (base-greedy NLL drift under Theta).

Update: Theta += (alpha/N) * sum_i z_i * eps_i   (DIGESTED form: the 1/sigma is folded into
alpha, matching the reference paper. alpha is therefore a step size on UNIT-noise directions
and is INDEPENDENT of sigma -- the two must be tuned separately. Members = Theta + sigma*eps_i,
so sigma controls exploration radius only and does NOT rescale the step.)

Corrected 2026-07-27 (Phase 0 audit, Defect A): this docstring previously claimed
(alpha/(N*sigma)), contradicting the code below. The MATH is correct and unchanged; only this
comment was wrong. Consequence for the record: the A1 sigma-sweep held alpha=2e-3 fixed while
sigma varied 25x, so it never varied the step size -- A1's negative result is void.
See track_b/PHASE0_CONFIG_AUDIT.md.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")  # Instruct (headroom on GSM8K), matches A2 3B-Instruct family
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--population_size", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=300)
    p.add_argument("--sigma", type=float, default=0.02)      # adapter exploration radius (SWEPT)
    p.add_argument("--alpha", type=float, default=2e-3)      # ES step size (scaled for ~9.4M adapter params)
    p.add_argument("--pop_seed", type=int, default=0)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_cap", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_prefix", required=True)
    # D2/Step-2 KL-velocity probe: measure KL drift only, skip the expensive ID + OOD passes.
    p.add_argument("--kl_only", action="store_true",
                   help="skip ID-accuracy generation at eval points (KL proxy only)")
    p.add_argument("--no_final_eval", action="store_true",
                   help="skip the final OOD/ID battery (short tuning probes)")
    return p.parse_args()


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

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from es_bench.shared_reward import math_reward
    from es_bench import data_math
    import lora_cfg
    import config as C

    N, B = a.population_size, a.batch
    tmpl = lora_cfg.build_template(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    _levels = [3, 4, 5] if a.dataset == "math" else None
    train, val = data_math.make_split(a.dataset, 2000, 300, a.data_seed, levels=_levels)
    import random
    drng = random.Random(a.data_seed)
    step_batches = [[drng.randrange(len(train)) for _ in range(B)] for _ in range(a.num_steps)]
    eval_rows = val[:a.eval_cap]
    eval_prompts = [data_math.build_prompt(tok, r["question"]) for r in eval_rows]
    eval_gts = [r["gt"] for r in eval_rows]

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=C.TRAIN_GPU_MEM_UTIL, max_model_len=2048,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
              enable_lora=True, max_loras=N, max_lora_rank=lora_cfg.LORA_RANK, max_cpu_loras=2 * N + 8,
              worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA, lora_cfg.TARGET_MODULES, 0))
    THETA_ID = 990000

    def eval_theta(step):
        tid = THETA_ID + step
        llm.collective_rpc("es_lora_inject_theta", args=(tid,))
        req = LoRARequest(f"theta{step}", tid, "/es/inmem")
        acc = None
        if not a.kl_only:
            sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
            outs = llm.generate(eval_prompts, sp, lora_request=[req] * len(eval_prompts), use_tqdm=False)
            acc = float(np.mean([math_reward(o.outputs[0].text, gt)["reward"] for o, gt in zip(outs, eval_gts)]))
        # KL proxy: NLL increase of base-greedy completions under Theta
        kl = None
        if _kl_base:
            sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
            reqs = [TokensPrompt(prompt_token_ids=r["ids"]) for r in _kl_base]
            o2 = llm.generate(reqs, sp2, lora_request=[req] * len(_kl_base), use_tqdm=False)
            num = den = 0.0
            for r, out in zip(_kl_base, o2):
                plp = out.prompt_logprobs
                for pos in range(r["a"], r["b"]):
                    e = plp[pos] if pos < len(plp) else None
                    if e and r["ids"][pos] in e:
                        num += (r["lp"][pos - r["a"]] - e[r["ids"][pos]].logprob); den += 1
            kl = float(num / den) if den else 0.0
        return acc, kl

    # capture base-greedy completions (no adapter) for the KL proxy
    _kl_base = []
    klp = [data_math.build_prompt(tok, val[a.eval_cap + i]["question"]) for i in range(min(100, len(val) - a.eval_cap))]
    spb = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=0)
    for p, o in zip(klp, llm.generate(klp, spb, use_tqdm=False)):
        c = o.outputs[0]
        pid = tok(p, add_special_tokens=False)["input_ids"]
        _kl_base.append({"ids": pid + list(c.token_ids), "a": len(pid), "b": len(pid) + len(c.token_ids),
                         "lp": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)]})

    prng = np.random.RandomState(a.pop_seed)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jf = open(a.out_prefix + ".jsonl", "w")
    acc0, kl0 = eval_theta(0)
    curve = [{"step": 0, "id_acc": acc0, "kl": kl0}]
    _f = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"[lora-es s{a.pop_seed} sig{a.sigma} a{a.alpha}] step 0 ID={_f(acc0)} KL={kl0:.4f}", flush=True)
    t0 = time.perf_counter()

    for step in range(a.num_steps):
        batch = [train[i] for i in step_batches[step]]
        prompts = [data_math.build_prompt(tok, b["question"]) for b in batch]
        gts = [b["gt"] for b in batch]
        seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(N)]
        ids = llm.collective_rpc("es_lora_inject_members", args=(1 + step * N, seeds, a.sigma))[0]
        reqs, sps, lrs = [], [], []
        for i, lid in enumerate(ids):
            lr = LoRARequest(f"s{step}m{i}", lid, "/es/inmem")
            for b in range(B):
                reqs.append(prompts[b]); lrs.append(lr)
                sps.append(SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=seeds[i]))
        outs = llm.generate(reqs, sps, lora_request=lrs, use_tqdm=False)
        fit = np.array([np.mean([math_reward(outs[i * B + b].outputs[0].text, gts[b])["reward"] for b in range(B)])
                        for i in range(N)], dtype=np.float64)
        z = (fit - fit.mean()) / (fit.std() + 1e-8)
        # (alpha/N)*z on unit-noise dirs; sigma controls exploration only (digested form -- see
        # module docstring). NOTE: ||z||=sqrt(N) always, so ||Delta|| ~ alpha*sqrt(d/N) on every
        # non-zero step regardless of signal strength (Phase 0 audit, Defect B).
        coeffs = ((a.alpha / N) * z).tolist()
        upd_sq = llm.collective_rpc("es_lora_commit", args=(seeds, coeffs))[0]
        row = {"step": step, "fit_mean": float(fit.mean()), "fit_std": float(fit.std()),
               "zero_update": bool(fit.std() < 1e-12), "update_l2": math.sqrt(max(upd_sq, 0.0))}
        jf.write(json.dumps(row) + "\n"); jf.flush()
        if (step + 1) % a.eval_every == 0:
            acc, kl = eval_theta(step + 1)
            curve.append({"step": step + 1, "id_acc": acc, "kl": kl})
            print(f"[lora-es s{a.pop_seed} sig{a.sigma} a{a.alpha}] step {step+1} fit={fit.mean():.3f} "
                  f"ID={_f(acc)} KL={kl:.4f} |Δ|={row['update_l2']:.3f}", flush=True)
    jf.close()

    # final OOD/ID battery under the trained mean adapter Theta
    ood_final = None
    if not a.no_final_eval:
        import eval_core
        tidf = THETA_ID + 999999
        llm.collective_rpc("es_lora_inject_theta", args=(tidf,))
        theta_req = LoRARequest("theta_final", tidf, "/es/inmem")
        ood_final = eval_core.eval_on_llm(llm, tok, cap=300, max_tokens=2048, lora_request=theta_req)
        print("OOD_FINAL:", json.dumps(ood_final, indent=2), flush=True)

    _rows = [json.loads(l) for l in open(a.out_prefix + ".jsonl")]
    _fm = np.array([r["fit_mean"] for r in _rows])
    _w = max(1, min(20, len(_fm) // 2))
    summary = {"model": a.model, "population_size": N, "sigma": a.sigma, "alpha": a.alpha,
               "batch": B, "ood_final": ood_final,
               "num_steps": a.num_steps, "pop_seed": a.pop_seed, "id_curve": curve,
               "id_base": acc0, "id_final": curve[-1]["id_acc"],
               "id_gain": (None if (acc0 is None or curve[-1]["id_acc"] is None)
                           else curve[-1]["id_acc"] - acc0),
               # matched estimator: same window used for the ES/GRPO train-fitness comparison
               "fit_first_w": float(_fm[:_w].mean()), "fit_last_w": float(_fm[-_w:].mean()),
               "fit_delta": float(_fm[-_w:].mean() - _fm[:_w].mean()), "fit_window": _w,
               "kl_final": curve[-1]["kl"],
               "kl_velocity_per_step": (curve[-1]["kl"] / a.num_steps) if a.num_steps else None,
               "zero_update_rate": float(np.mean([r["zero_update"] for r in _rows])),
               "update_l2_mean_nonzero": float(np.mean([r["update_l2"] for r in _rows
                                                        if r["update_l2"] > 1e-12] or [0.0])),
               "wall_clock_s": time.perf_counter() - t0}
    json.dump(summary, open(a.out_prefix + "_summary.json", "w"), indent=2)
    print("SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "id_curve"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
