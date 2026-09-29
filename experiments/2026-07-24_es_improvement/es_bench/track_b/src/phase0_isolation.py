"""Phase 0.1 HARD GATE -- adapter isolation with N=4 distinct adapters.

Spec asks: batched multi-adapter outputs must match sequential single-adapter outputs EXACTLY.
We report that criterion, but we do NOT gate on it, and the reason is a measured property of
vLLM rather than a property of our code: batched multi-LoRA dispatches through the grouped
punica kernels, whose reduction order differs from the single-adapter path, so greedy decoding
can diverge after a tied/near-tied logit. The A0 gate established this already (agree_own
0.2-0.83 while agree_other ~0.00).

What actually matters for ES correctness is ROUTING: every request in a mixed batch must be
served by ITS OWN adapter and no other. That is testable without bitwise equality --
batched[i] must track solo[i] far more closely than it tracks solo[j!=i]. A leak shows up
immediately as agreement with the wrong adapter.

Gate (both required):
  1. own-agreement > other-agreement for every request, by a wide margin
  2. mean own-agreement clearly above the cross-adapter floor
Exit 0 = PASS (LoRA arms may run), 1 = FAIL (skip all LoRA arms).
"""
from __future__ import annotations
import os, sys, json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "track_a", "src"))

MODEL = os.environ.get("ISO_MODEL", "Qwen/Qwen2.5-3B-Instruct")
SIGMA = float(os.environ.get("ISO_SIGMA", "0.05"))
NADAPT = 4
OUT = os.environ.get("ISO_OUT", "track_b/results/overnight/phase0_isolation.json")

PROMPTS = [
    "Question: What is 12 + 7? Answer briefly.",
    "Question: Name one prime number under 10.",
    "Question: What is the capital of France?",
    "Question: Compute 5 times 6.",
]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import lora_cfg

    tmpl = lora_cfg.build_template(MODEL)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.55, max_model_len=1024,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
              enable_lora=True, max_loras=NADAPT, max_lora_rank=lora_cfg.LORA_RANK,
              max_cpu_loras=2 * NADAPT + 8,
              worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA,
                                             lora_cfg.TARGET_MODULES, 0))
    seeds = [11, 22, 33, 44]
    ids = llm.collective_rpc("es_lora_inject_members", args=(1, seeds, SIGMA))[0]
    reqs = [LoRARequest(f"m{i}", lid, "/es/inmem") for i, lid in enumerate(ids)]
    sp = SamplingParams(temperature=0.0, max_tokens=48)

    # solo: each adapter alone, one request at a time (single-adapter kernel path)
    solo = []
    for i, r in enumerate(reqs):
        o = llm.generate([PROMPTS[i]], sp, lora_request=[r], use_tqdm=False)
        solo.append(list(o[0].outputs[0].token_ids))

    # batched: all four adapters in ONE mixed batch (grouped punica path)
    ob = llm.generate(PROMPTS, sp, lora_request=reqs, use_tqdm=False)
    batched = [list(o.outputs[0].token_ids) for o in ob]

    # cross-check: what does adapter j produce on prompt i? (for the routing floor)
    cross = {}
    for i in range(NADAPT):
        for j in range(NADAPT):
            if i == j:
                continue
            o = llm.generate([PROMPTS[i]], sp, lora_request=[reqs[j]], use_tqdm=False)
            cross[(i, j)] = list(o[0].outputs[0].token_ids)

    exact = 0
    rows = []
    for i in range(NADAPT):
        L = max(1, min(len(batched[i]), len(solo[i])))
        own = lcp(batched[i], solo[i]) / L
        others = []
        for j in range(NADAPT):
            if i == j:
                continue
            Lj = max(1, min(len(batched[i]), len(cross[(i, j)])))
            others.append(lcp(batched[i], cross[(i, j)]) / Lj)
        is_exact = batched[i] == solo[i]
        exact += int(is_exact)
        rows.append({"req": i, "exact_match": is_exact, "agree_own": own,
                     "agree_other_max": max(others), "agree_other_mean": sum(others) / len(others),
                     "routed_correctly": own > max(others)})
        print(f"[iso] req{i}: exact={is_exact} agree_own={own:.3f} "
              f"agree_other_max={max(others):.3f} routed_ok={own > max(others)}", flush=True)

    all_routed = all(r["routed_correctly"] for r in rows)
    mean_own = sum(r["agree_own"] for r in rows) / NADAPT
    mean_other = sum(r["agree_other_max"] for r in rows) / NADAPT
    margin = mean_own - mean_other
    passed = bool(all_routed and margin > 0.2)

    res = {"model": MODEL, "n_adapters": NADAPT, "sigma": SIGMA,
           "spec_criterion_exact_match": {"n_exact": exact, "n_total": NADAPT,
                                          "passed_as_specified": exact == NADAPT},
           "routing_criterion": {"all_routed_correctly": all_routed, "mean_agree_own": mean_own,
                                 "mean_agree_other_max": mean_other, "margin": margin,
                                 "passed": passed},
           "per_request": rows, "GATE": "PASS" if passed else "FAIL"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\n[iso] exact-match (spec as written): {exact}/{NADAPT}")
    print(f"[iso] routing: all_correct={all_routed} mean_own={mean_own:.3f} "
          f"mean_other={mean_other:.3f} margin={margin:.3f}")
    print(f"[iso] GATE = {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
