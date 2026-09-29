#!/usr/bin/env python
"""D2 -- perturbation-scale probe (NO training).

Calibrates the LoRA exploration radius sigma_lora against the full-param reference
sigma=1e-3 on a BEHAVIORAL scale, using two matched observables on a fixed prompt set:

  flip_rate : fraction of prompts whose extracted final answer changes under the
              perturbation (vs the unperturbed base answer). Pure behavior, no logits.
  kl_drift  : mean per-token NLL increase of the BASE greedy completion under the
              perturbed weights (track_b/src/kl.py proxy). Same estimator both arms.

Arms (run separately -- one vLLM instance each, since the worker extension differs):
  --arm fullparam : theta = base + sigma*eps   (ESWorker.es_set_member), solo per seed
  --arm lora      : Theta0 + sigma*eps as LoRA adapters (ESLoRAWorker), ONE batched
                    multi-adapter generate -- matching how members are actually scored
                    during LoRA-ES training (A0: batched multi-LoRA differs numerically
                    from solo, so calibrating in the batched regime is the correct choice).

sigma* = the sigma_lora whose (flip_rate, kl_drift) pair matches the full-param
sigma=1e-3 reference. Reported by combine_probe.py, not here.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["fullparam", "lora"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="math")
    p.add_argument("--n_prompts", type=int, default=32)
    p.add_argument("--n_seeds", type=int, default=8, help="perturbation draws per sigma")
    p.add_argument("--sigmas", default=None, help="comma list; default per-arm grid")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--data_seed", type=int, default=1234)
    p.add_argument("--probe_seed", type=int, default=7)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=None, help="LoRA arm: rank override")
    p.add_argument("--targets", default=None, help="LoRA arm: comma list of target modules")
    return p.parse_args()


DEFAULT_SIGMAS = {
    "fullparam": [3e-4, 1e-3, 3e-3],                       # 1e-3 = the reference point
    "lora": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],    # log grid to bracket the match
}


def main():
    a = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, HERE)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import extract_final_answer, normalize_answer
    from es_bench import data_math
    import kl as klmod

    sigmas = [float(x) for x in a.sigmas.split(",")] if a.sigmas else DEFAULT_SIGMAS[a.arm]
    levels = [3, 4, 5] if a.dataset == "math" else None
    _, val = data_math.make_split(a.dataset, 2000, 300, a.data_seed, levels=levels)
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = val[:a.n_prompts]
    prompts = [data_math.build_prompt(tok, r["question"]) for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    def answers(texts):
        out = []
        for t in texts:
            e = extract_final_answer(t)
            out.append(normalize_answer(str(e.extracted)) if e.success else None)
        return out

    def flip_rate(base_ans, new_ans):
        return float(np.mean([int(b != n) for b, n in zip(base_ans, new_ans)]))

    prng = np.random.RandomState(a.probe_seed)
    seeds = [int(prng.randint(1, 2**31 - 1)) for _ in range(a.n_seeds)]
    results = {"arm": a.arm, "model": a.model, "dataset": a.dataset,
               "n_prompts": len(prompts), "n_seeds": a.n_seeds, "seeds": seeds,
               "max_tokens": a.max_tokens, "sigmas": sigmas, "points": []}
    t_run0 = time.perf_counter()

    # ---------------------------------------------------------------- full-param arm
    if a.arm == "fullparam":
        llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.85,
                  max_model_len=2048, enforce_eager=False, disable_log_stats=True,
                  enable_prefix_caching=False,
                  worker_extension_cls="es_bench.track_b.src.phase4_worker.Phase4Worker")
        llm.collective_rpc("es_snapshot_base")
        base_texts = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
        base_ans = answers(base_texts)
        rec = klmod.capture_base(llm, prompts, max_tokens=a.max_tokens)
        self_drift = klmod.drift(llm, tok, rec)
        print(f"[sanity] base-vs-base drift = {self_drift:.3e} (expect ~0)", flush=True)
        results["base_extract_rate"] = float(np.mean([x is not None for x in base_ans]))
        results["sanity_self_drift"] = self_drift

        for sig in sigmas:
            fl, kls = [], []
            for s in seeds:
                llm.collective_rpc("es_set_member", args=(s, sig))
                texts = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
                fl.append(flip_rate(base_ans, answers(texts)))
                kls.append(klmod.drift(llm, tok, rec))
                llm.collective_rpc("es_restore_base")
            results["points"].append({
                "sigma": sig, "flip_rate_mean": float(np.mean(fl)), "flip_rate_std": float(np.std(fl)),
                "kl_drift_mean": float(np.mean(kls)), "kl_drift_std": float(np.std(kls)),
                "flip_rates": fl, "kl_drifts": kls})
            print(f"[fullparam] sigma={sig:.1e}  flip={np.mean(fl):.3f}+-{np.std(fl):.3f}  "
                  f"KL={np.mean(kls):.4e}", flush=True)

    # ---------------------------------------------------------------- LoRA arm
    else:
        from vllm import TokensPrompt
        from vllm.lora.request import LoRARequest
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "track_a", "src"))
        import lora_cfg

        rank = a.rank if a.rank is not None else lora_cfg.LORA_RANK
        targets = a.targets.split(",") if a.targets else lora_cfg.TARGET_MODULES
        results["rank"], results["targets"] = rank, list(targets)
        K = a.n_seeds
        llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.75,
                  max_model_len=2048, enforce_eager=False, disable_log_stats=True,
                  enable_prefix_caching=False, enable_lora=True, max_loras=K,
                  max_lora_rank=rank, max_cpu_loras=2 * K + 8,
                  worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
        tmpl = lora_cfg.build_template(a.model, rank=rank, targets=targets)
        llm.collective_rpc("es_lora_init", args=(tmpl, rank, lora_cfg.LORA_ALPHA,
                                                 targets, 0))
        base_texts = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
        base_ans = answers(base_texts)
        rec = klmod.capture_base(llm, prompts, max_tokens=a.max_tokens)
        results["base_extract_rate"] = float(np.mean([x is not None for x in base_ans]))

        # teacher-forced NLL of the base-greedy completions under a given adapter
        pre_ids = [tok(r["prompt"], add_special_tokens=False)["input_ids"] for r in rec]
        full_ids = [p + r["comp_ids"] for p, r in zip(pre_ids, rec)]
        spans = [(len(p), len(f)) for p, f in zip(pre_ids, full_ids)]
        sp_tf = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

        def lora_drift(req):
            outs = llm.generate([TokensPrompt(prompt_token_ids=f) for f in full_ids], sp_tf,
                                lora_request=[req] * len(full_ids), use_tqdm=False)
            num = den = 0.0
            for out, (s0, s1), f, r in zip(outs, spans, full_ids, rec):
                plp = out.prompt_logprobs
                for pos in range(s0, s1):
                    e = plp[pos] if pos < len(plp) else None
                    if e and f[pos] in e:
                        num += (r["base_lp"][pos - s0] - e[f[pos]].logprob); den += 1
            return float(num / den) if den else 0.0

        lid = [100]

        def next_id():
            lid[0] += 1
            return lid[0]

        # sanity: Theta0 has B=0 -> the adapter is an exact no-op; must reproduce base
        tid = next_id()
        llm.collective_rpc("es_lora_inject_theta", args=(tid,))
        req0 = LoRARequest("theta0", tid, "/es/inmem")
        t0_texts = [o.outputs[0].text for o in
                    llm.generate(prompts, sp, lora_request=[req0] * len(prompts), use_tqdm=False)]
        results["sanity_theta0_flip"] = flip_rate(base_ans, answers(t0_texts))
        results["sanity_theta0_drift"] = lora_drift(req0)
        print(f"[sanity] Theta0(B=0) flip={results['sanity_theta0_flip']:.3f} "
              f"drift={results['sanity_theta0_drift']:.3e} (expect ~0/~0)", flush=True)

        for sig in sigmas:
            # ONE batched multi-adapter generate over all K perturbation draws -- the
            # same regime used to score members during training.
            ids = llm.collective_rpc("es_lora_inject_members",
                                     args=(next_id() * 1000, seeds, sig))[0]
            reqs = [LoRARequest(f"s{sig}m{i}", int(l), "/es/inmem") for i, l in enumerate(ids)]
            batch_p, batch_r = [], []
            for r_ in reqs:
                batch_p.extend(prompts); batch_r.extend([r_] * len(prompts))
            outs = llm.generate(batch_p, sp, lora_request=batch_r, use_tqdm=False)
            P = len(prompts)
            fl = [flip_rate(base_ans, answers([outs[i * P + j].outputs[0].text for j in range(P)]))
                  for i in range(K)]
            kls = [lora_drift(r_) for r_ in reqs]
            results["points"].append({
                "sigma": sig, "flip_rate_mean": float(np.mean(fl)), "flip_rate_std": float(np.std(fl)),
                "kl_drift_mean": float(np.mean(kls)), "kl_drift_std": float(np.std(kls)),
                "flip_rates": fl, "kl_drifts": kls})
            print(f"[lora] sigma={sig:.1e}  flip={np.mean(fl):.3f}+-{np.std(fl):.3f}  "
                  f"KL={np.mean(kls):.4e}", flush=True)

    results["wall_clock_s"] = time.perf_counter() - t_run0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
