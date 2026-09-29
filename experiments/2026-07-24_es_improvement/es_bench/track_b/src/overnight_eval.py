#!/usr/bin/env python
"""Phase D -- evaluation battery for the overnight run.

Differs from eval_core.eval_on_llm in three ways the spec requires:
  * sampling rule: FULL set if n <= 1500, else a fixed-seed 500-row random subset
    (eval_core takes a deterministic prefix, which is not the same thing)
  * two extra OOD sets: ASDiv and GSM-Symbolic (the GSM8K shift axis)
  * capability window: OOD sets whose BASE accuracy falls outside [0.05, 0.85] are dropped
    from the OOD averages (too saturated or too floored to carry signal), and listed.

One process evaluates everything so the 3B base is loaded once: base, then each arm's adapter
checkpoints, then the shrinkage frontier lambda*Delta for lambda in {0.25,0.5,0.75,1.0}
(lambda=1.0 IS the final checkpoint, so it is not evaluated twice).

Results are written after EVERY eval, so a deadline cut still leaves usable partial output.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path

CAP_FULL_MAX = 1500
SUBSET_N = 500
SUBSET_SEED = 12345
WINDOW = (0.05, 0.85)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--spec", required=True, help="JSON: [{tag,arm,ckpt,lambdas}]")
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--kl_probe_n", type=int, default=64)
    p.add_argument("--deadline_ts", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


# arm -> (ID set, OOD sets, probe)
ARM_SETS = {
    "gsm8k": {"id": ["gsm8k"],
              "ood": ["svamp", "asdiv", "gsm_symbolic", "math500", "minerva_math"],
              "probe": ["countdown"]},
    "math": {"id": ["math500"],
             "ood": ["gsm8k", "svamp", "minerva_math", "olympiadbench"],
             "probe": ["countdown"]},
}
ALL_SETS = ["gsm8k", "svamp", "asdiv", "gsm_symbolic", "math500", "minerva_math",
            "olympiadbench", "countdown"]


def extra_adapters():
    def asdiv(r):
        q = (r["body"].rstrip() + " " + r["question"]).strip()
        ans = str(r["answer"]).split("(")[0].strip()      # "9 (apples)" -> "9"
        return (q, ans, None) if ans else None

    def gsm_symbolic(r):
        from es_bench.data_math import _gsm_gt
        return r["question"], _gsm_gt(r["answer"]), None

    return {"asdiv": (("EleutherAI/asdiv", None, "validation"), asdiv),
            "gsm_symbolic": (("apple/GSM-Symbolic", "main", "test"), gsm_symbolic)}


def get_rows(name):
    """Spec sampling rule: full if n<=1500 else fixed-seed 500 subset."""
    import eval_core
    ad = dict(eval_core.adapters())
    ad.update(extra_adapters())
    spec, fn = ad[name]
    rows, _, full_n = eval_core.load_rows(spec, fn, None)
    n = len(rows)
    if n > CAP_FULL_MAX:
        rnd = random.Random(SUBSET_SEED)
        rows = rnd.sample(rows, SUBSET_N)
        return rows, full_n, True
    return rows, full_n, False


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
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    from es_bench.shared_reward import math_reward
    from es_bench import data_math
    import eval_core
    import lora_cfg

    todo = json.load(open(a.spec))
    tok = AutoTokenizer.from_pretrained(a.model)
    tmpl = lora_cfg.build_template(a.model)

    needed = sorted({s for t in todo for s in
                     (ARM_SETS[t["arm"]]["id"] + ARM_SETS[t["arm"]]["ood"] +
                      ARM_SETS[t["arm"]]["probe"])})
    print(f"[eval] datasets needed: {needed}", flush=True)
    DATA = {}
    for name in needed:
        if name == "countdown":
            continue
        rows, full_n, sub = get_rows(name)
        DATA[name] = {"rows": rows, "full_n": full_n, "subsampled": sub}
        print(f"[data] {name}: n={len(rows)} (full {full_n}) subsampled={sub}", flush=True)

    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.80, max_model_len=2048,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True,
              enable_lora=True, max_loras=2, max_lora_rank=lora_cfg.LORA_RANK, max_cpu_loras=8,
              worker_extension_cls="es_bench.track_b.src.lora_main_worker.LoRAMainWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA,
                                             lora_cfg.TARGET_MODULES, 0))

    # ---- fixed KL probe (base-greedy capture, no adapter) ----
    _, val = data_math.make_split("math", 2000, 300, 1234, levels=[3, 4, 5])
    klp = [data_math.build_prompt(tok, val[300 - a.kl_probe_n + i]["question"])
           for i in range(a.kl_probe_n)]
    spb = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=0)
    kl_base = []
    for p, o in zip(klp, llm.generate(klp, spb, use_tqdm=False)):
        c = o.outputs[0]
        pid = tok(p, add_special_tokens=False)["input_ids"]
        kl_base.append({"ids": pid + list(c.token_ids), "a": len(pid),
                        "b": len(pid) + len(c.token_ids),
                        "lp": [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)]})

    def kl_of(req):
        sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        reqs = [TokensPrompt(prompt_token_ids=r["ids"]) for r in kl_base]
        lr = [req] * len(kl_base) if req is not None else None
        outs = llm.generate(reqs, sp2, lora_request=lr, use_tqdm=False)
        num = den = 0.0
        for r, out in zip(kl_base, outs):
            plp = out.prompt_logprobs
            for pos in range(r["a"], r["b"]):
                e = plp[pos] if pos < len(plp) else None
                if e and r["ids"][pos] in e:
                    num += (r["lp"][pos - r["a"]] - e[r["ids"][pos]].logprob)
                    den += 1
        return float(num / den) if den else 0.0

    def eval_set(name, req):
        if name == "countdown":
            return eval_core._eval_countdown(llm, tok, SUBSET_N, a.max_tokens, None, req)
        d = DATA[name]
        rows = d["rows"]
        prompts = [data_math.build_prompt(tok, r["q"]) for r in rows]
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
        lr = [req] * len(prompts) if req is not None else None
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
        gen_s = time.perf_counter() - t0
        nc = ne = ntr = 0
        per_level = {}
        for r, o in zip(rows, outs):
            op = o.outputs[0]
            res = math_reward(op.text, r["gold"])
            c = int(res["reward"] >= 1.0)
            nc += c
            ne += int(res["reward_info"]["extracted"])
            ntr += int(op.finish_reason == "length")
            if r["level"] is not None:
                pl = per_level.setdefault(int(r["level"]), [0, 0])
                pl[0] += c
                pl[1] += 1
        n = len(rows)
        out = {"n": n, "full_n": d["full_n"], "subsampled": d["subsampled"],
               "accuracy": round(nc / n, 4), "extract_rate": round(ne / n, 4),
               "trunc_rate": round(ntr / n, 4), "gen_seconds": round(gen_s, 1)}
        if per_level:
            out["per_level_accuracy"] = {k: round(v[0] / v[1], 4) for k, v in sorted(per_level.items())}
            l12 = [v for k, v in per_level.items() if k <= 2]
            l35 = [v for k, v in per_level.items() if k >= 3]
            if l12:
                out["L1_2_accuracy"] = round(sum(x[0] for x in l12) / sum(x[1] for x in l12), 4)
            if l35:
                out["L3_5_accuracy"] = round(sum(x[0] for x in l35) / sum(x[1] for x in l35), 4)
        return out

    results = {"model": a.model, "protocol": {
        "greedy": True, "max_tokens": a.max_tokens, "cap_rule": f"full if n<={CAP_FULL_MAX} else "
        f"{SUBSET_N}-row subset (seed {SUBSET_SEED})", "capability_window": list(WINDOW),
        "kl_probe_n": a.kl_probe_n}, "entries": {}}

    def flush():
        json.dump(results, open(a.out, "w"), indent=2)

    # ---- base ----
    t0 = time.perf_counter()
    base_ev = {s: eval_set(s, None) for s in needed}
    results["entries"]["base"] = {"arm": "base", "lambda": 0.0, "kl": kl_of(None),
                                  "eval": base_ev, "seconds": time.perf_counter() - t0}
    flush()
    print("[eval] base done: " + ", ".join(f"{k}={v['accuracy']}" for k, v in base_ev.items()),
          flush=True)

    dropped = {s: base_ev[s]["accuracy"] for s in needed
               if not (WINDOW[0] <= base_ev[s]["accuracy"] <= WINDOW[1])}
    results["capability_window_dropped"] = dropped
    flush()
    print(f"[eval] capability-window violators (base acc outside {WINDOW}): {dropped}", flush=True)

    THETA_ID = 700000
    nid = [0]

    for t in todo:
        if a.deadline_ts and time.time() > a.deadline_ts:
            results["stopped_early"] = f"deadline before {t['tag']}"
            flush()
            print(f"[eval] DEADLINE -- skipping remaining entries from {t['tag']}", flush=True)
            break
        ck = t.get("ckpt")
        if not ck or not os.path.exists(ck):
            print(f"[eval] SKIP {t['tag']}: checkpoint missing ({ck})", flush=True)
            results["entries"][t["tag"]] = {"error": f"checkpoint missing: {ck}"}
            flush()
            continue
        state = torch.load(ck, map_location="cpu")
        llm.collective_rpc("es_lora_load_theta", args=(state,))
        sets = (ARM_SETS[t["arm"]]["id"] + ARM_SETS[t["arm"]]["ood"] + ARM_SETS[t["arm"]]["probe"])
        for lam in t.get("lambdas", [1.0]):
            if a.deadline_ts and time.time() > a.deadline_ts:
                results["stopped_early"] = f"deadline inside {t['tag']}"
                flush()
                break
            nid[0] += 1
            lid = THETA_ID + nid[0]
            llm.collective_rpc("es_lora_inject_scaled", args=(lid, lam))
            req = LoRARequest(f"{t['tag']}_l{lam}", lid, "/es/inmem")
            t1 = time.perf_counter()
            ev = {s: eval_set(s, req) for s in sets}
            tag = f"{t['tag']}_lambda{lam:g}"
            results["entries"][tag] = {"arm": t["arm"], "lambda": lam, "ckpt": ck,
                                       "kl": kl_of(req), "eval": ev,
                                       "seconds": time.perf_counter() - t1}
            flush()
            print(f"[eval] {tag}: " + ", ".join(f"{k}={v['accuracy']}" for k, v in ev.items()),
                  flush=True)

    flush()
    print(f"[eval] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
