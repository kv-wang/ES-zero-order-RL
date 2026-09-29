#!/usr/bin/env python
"""Phase 1 D2: perturbation-scale probe (no training). On 32 fixed L3-5 prompts, measure per-member
answer FLIP-RATE (vs base greedy) and mean per-token NLL drift (teacher-forced on base-greedy tokens).
  --mode full  : full-param ES at fixed sigma (reference behavioral scale).
  --mode lora  : LoRA-ES over a sigma log-grid -> curve; pick sigma* matching the full-param reference.
"""
import argparse, json, os, re, sys
import numpy as np

def extract_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; depth = 1; out = []
    while j < len(t) and depth:
        c = t[j]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        if depth: out.append(c)
        j += 1
    return "".join(out).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sigmas", default="1e-3")      # comma list; full: usually just 1e-3
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "src"))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "track_a", "src"))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from es_bench.data_math import build_prompt, make_split
    import kl as klmod
    import pandas as pd

    tok = AutoTokenizer.from_pretrained(a.model)
    # 32 fixed L3-5 prompts (deterministic: GRPO val set, same rows every run)
    df = pd.read_parquet(os.path.join(HERE, "..", "grpo_ood", "data_math", "val.parquet")).iloc[:a.n_prompts]
    prompts = [build_prompt(tok, list(r["prompt"])[0]["content"]) for _, r in df.iterrows()]
    sigmas = [float(x) for x in a.sigmas.split(",")]

    def nll_drift(rec, lora_request=None):
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        reqs, spans, fulls = [], [], []
        for r in rec:
            pid = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            full = pid + r["comp_ids"]; reqs.append(TokensPrompt(prompt_token_ids=full))
            spans.append((len(pid), len(full))); fulls.append(full)
        lr = ([lora_request] * len(reqs)) if lora_request is not None else None
        outs = llm.generate(reqs, sp, lora_request=lr, use_tqdm=False)
        num = den = 0.0
        for out, (s, e), full, r in zip(outs, spans, fulls, rec):
            plp = out.prompt_logprobs
            for k, pos in enumerate(range(s, e)):
                ent = plp[pos] if pos < len(plp) else None
                if ent and full[pos] in ent:
                    num += (r["base_lp"][k] - ent[full[pos]].logprob); den += 1
        return float(num / den) if den else 0.0

    def greedy_answers(lora_request=None):
        sp = SamplingParams(temperature=0.0, max_tokens=512)
        lr = ([lora_request] * len(prompts)) if lora_request is not None else None
        outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
        return [extract_boxed(o.outputs[0].text) for o in outs]

    if a.mode == "full":
        llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.6, max_model_len=2048,
                  enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
                  worker_extension_cls="es_bench.track_b.src.phase4_worker.Phase4Worker")
        llm.collective_rpc("es_snapshot_base")
        base_ans = greedy_answers()
        rec = klmod.capture_base(llm, prompts, max_tokens=512)
        prng = np.random.RandomState(0)
        curve = []
        for sg in sigmas:
            flips, kls = [], []
            for _ in range(a.seeds):
                seed = int(prng.randint(1, 2**31 - 1))
                llm.collective_rpc("es_set_member", args=(seed, sg))
                ans = greedy_answers()
                flips.append(float(np.mean([ans[i] != base_ans[i] for i in range(len(ans))])))
                kls.append(nll_drift(rec))
                llm.collective_rpc("es_restore_base")
            curve.append({"sigma": sg, "flip_rate": round(float(np.mean(flips)), 4),
                          "flip_sd": round(float(np.std(flips)), 4),
                          "nll_drift": round(float(np.mean(kls)), 6)})
            print("FULL", curve[-1], flush=True)
    else:
        import lora_cfg
        N_MAX = a.seeds
        llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.6, max_model_len=2048,
                  enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
                  enable_lora=True, max_loras=1, max_lora_rank=lora_cfg.LORA_RANK, max_cpu_loras=4,
                  worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
        tmpl = lora_cfg.build_template(a.model)
        llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA, lora_cfg.TARGET_MODULES, 0))
        base_ans = greedy_answers()                      # Theta=0 -> base
        rec = klmod.capture_base(llm, prompts, max_tokens=512)
        prng = np.random.RandomState(0)
        curve = []; lid = 1
        for sg in sigmas:
            flips, kls = [], []
            for _ in range(a.seeds):
                seed = int(prng.randint(1, 2**31 - 1))
                ids = llm.collective_rpc("es_lora_inject_members", args=(lid, [seed], sg))[0]
                req = LoRARequest(f"m{lid}", ids[0], "/es/inmem"); lid += 1
                ans = greedy_answers(req)
                flips.append(float(np.mean([ans[i] != base_ans[i] for i in range(len(ans))])))
                kls.append(nll_drift(rec, req))
            curve.append({"sigma": sg, "flip_rate": round(float(np.mean(flips)), 4),
                          "flip_sd": round(float(np.std(flips)), 4),
                          "nll_drift": round(float(np.mean(kls)), 6)})
            print("LORA", curve[-1], flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"mode": a.mode, "model": a.model, "n_prompts": len(prompts), "seeds": a.seeds,
               "curve": curve}, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
