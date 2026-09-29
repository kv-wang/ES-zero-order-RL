#!/usr/bin/env python
"""Measure mean generated tokens/sequence vs max_tokens cap, and validate a smaller-d LoRA config.

Runs safely alongside another job on the same GPU: the quantity it measures (L_bar = tokens per
sequence) is a property of the model and the cap, not of GPU contention. Wall-clock printed here
is contaminated by any co-tenant and must NOT be used -- combine L_bar with the fitted throughput
law instead:   t_step ~ (N*B)^0.397 * L_bar.

Also injects an ES-perturbed adapter under a reduced-d config (e.g. attn-only rank 8) to confirm
the injection path works and to report the perturbation norm sigma*sqrt(d), which is what has to
be re-calibrated when d changes.
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--caps", default="512,1024,2048")
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--sigma", type=float, default=0.015)
    ap.add_argument("--gpu_frac", type=float, default=0.15)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
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

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from es_bench import data_math
    from es_bench.shared_reward import math_reward
    import lora_cfg

    targets = a.targets.split(",")
    caps = [int(c) for c in a.caps.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    train, _ = data_math.make_split("math", 2000, 300, 1234, levels=[3, 4, 5])
    rows = train[: a.n_prompts]
    prompts = [data_math.build_prompt(tok, r["question"]) for r in rows]
    golds = [r["gt"] for r in rows]

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=a.gpu_frac,
              max_model_len=max(caps) + 1024, enforce_eager=False, enable_prefix_caching=False,
              disable_log_stats=True, enable_lora=True, max_loras=2, max_lora_rank=a.rank,
              max_cpu_loras=4,
              worker_extension_cls="es_bench.track_b.src.lora_main_worker.LoRAMainWorker")

    res = {"model": a.model, "n_prompts": a.n_prompts, "rank": a.rank, "targets": targets,
           "sigma": a.sigma, "note": "wall-clock contaminated by co-tenant; use L_bar only",
           "caps": {}}

    def run(cap, req=None, tag="base"):
        sp = SamplingParams(temperature=0.0, max_tokens=cap)
        lr = [req] * len(prompts) if req is not None else None
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
        el = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        ntr = sum(1 for o in outs if o.outputs[0].finish_reason == "length")
        nc = sum(1 for o, g in zip(outs, golds) if math_reward(o.outputs[0].text, g)["reward"] >= 1.0)
        r = {"L_bar": ntok / len(outs), "total_tokens": ntok, "trunc_rate": ntr / len(outs),
             "accuracy": nc / len(outs), "seconds_CONTAMINATED": round(el, 1)}
        print(f"[{tag}] cap={cap:5d}  L_bar={r['L_bar']:7.1f}  trunc={r['trunc_rate']:.3f}  "
              f"acc={r['accuracy']:.3f}  ({el:.1f}s, contaminated)", flush=True)
        return r

    for cap in caps:
        res["caps"][str(cap)] = {"base": run(cap, None, "base")}
        json.dump(res, open(a.out, "w"), indent=2)

    # ---- reduced-d adapter: init, perturb, verify it changes behaviour ----
    tmpl = lora_cfg.build_template(a.model, rank=a.rank, targets=targets)
    llm.collective_rpc("es_lora_init", args=(tmpl, a.rank, lora_cfg.LORA_ALPHA, targets, 0))
    # build_template returns a LIST of {shapeA,shapeB} dicts -- d is the trainable adapter
    # dimension, which sets both the ES sample complexity (d/N) and the perturbation norm
    # ||sigma*eps|| ~ sigma*sqrt(d) that has to be re-calibrated whenever d changes.
    d = sum(t["shapeA"][0] * t["shapeA"][1] + t["shapeB"][0] * t["shapeB"][1] for t in tmpl)
    ids = llm.collective_rpc("es_lora_inject_members", args=(9_000_001, [12345], a.sigma))[0]
    req = LoRARequest("probe_member", ids[0], "/es/inmem")
    mid = caps[len(caps) // 2]
    res["reduced_d"] = {"d_from_template": d, "n_modules": len(tmpl),
                        "perturb_norm_sigma_sqrt_d": a.sigma * (d ** 0.5), "lora_ids": ids,
                        "member": run(mid, req, f"r{a.rank}-{'+'.join(t[0] for t in targets)}")}
    res["reduced_d"]["base_same_cap"] = res["caps"][str(mid)]["base"]
    json.dump(res, open(a.out, "w"), indent=2)

    for c in caps:
        b = res["caps"][str(c)]["base"]["L_bar"]
        base512 = res["caps"][str(caps[0])]["base"]["L_bar"]
        print(f"L_bar ratio cap{c}/cap{caps[0]} = {b / base512:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
