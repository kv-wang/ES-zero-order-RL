"""Shared eval battery: greedy pass@1 + house extraction/reward on pinned sets.

eval_on_llm() runs against an ALREADY-LOADED vLLM model, so the trainer can score
its final (in-memory) trained weights with no checkpoint->HF conversion (vLLM fuses
param names). Used by both es_train_continuous.py (in-process) and eval_model.py (CLI).
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def adapters():
    import config as C

    def math500(r):   return r["problem"], r["answer"], int(r["level"])
    def gsm8k(r):
        from es_bench.data_math import _gsm_gt
        return r["question"], _gsm_gt(r["answer"]), None
    def svamp(r):     return (r["Body"].rstrip() + " " + r["Question"]).strip(), str(r["Answer"]), None
    def minerva(r):   return r["question"], str(r["answer"]), None
    def olympiad(r):
        if any(r.get(f"image_{i}") for i in range(1, 10)):
            return None
        fa = r["final_answer"]
        gold = (fa[0] if isinstance(fa, list) and fa else fa)
        return (r["question"], str(gold).strip(), None) if gold is not None else None
    def amc23(r):     return r["question"], str(r["answer"]), None

    return {"math500": (C.EVAL_DATASETS["math500"], math500),
            "gsm8k": (C.EVAL_DATASETS["gsm8k"], gsm8k),
            "svamp": (C.EVAL_DATASETS["svamp"], svamp),
            "minerva_math": (C.EVAL_DATASETS["minerva_math"], minerva),
            "olympiadbench": (C.EVAL_DATASETS["olympiadbench"], olympiad),
            "amc23": (C.EVAL_DATASETS["amc23"], amc23)}


# Countdown OOD probe: numbers->target equation game (repo-root task + verifier).
COUNTDOWN_JSON = "/home/hyin66/es-fine-tuning-paper/countdown/data/countdown.json"
COUNTDOWN_TASK_DIR = "/home/hyin66/es-fine-tuning-paper/countdown"


def _eval_countdown(llm, tok, cap, max_tokens, perq_prefix, lora_request):
    """Greedy pass@1 on Countdown. Correct == last <answer> uses each number once and
    evaluates to target (countdown_task.answer_reward_function). Prompt = the dataset's
    own `context` (raw completion, opens with <think>), matching the original ES setup."""
    from vllm import SamplingParams
    if COUNTDOWN_TASK_DIR not in sys.path:
        sys.path.insert(0, COUNTDOWN_TASK_DIR)
    from countdown_task import answer_reward_function
    data = json.load(open(COUNTDOWN_JSON))
    rows = data[:cap]                              # pinned deterministic slice
    prompts = [r["context"] for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    lr = ([lora_request] * len(prompts)) if lora_request is not None else None
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
    gen_s = time.perf_counter() - t0
    n_correct = n_extract = 0
    perq = open(f"{perq_prefix}__countdown_perq.jsonl", "w") if perq_prefix else None
    for r, o in zip(rows, outs):
        txt = o.outputs[0].text
        c = int(answer_reward_function(txt, r["numbers"], r["target"]) >= 1.0)
        ext = int("<answer>" in txt and "</answer>" in txt)
        n_correct += c; n_extract += ext
        if perq:
            perq.write(json.dumps({"numbers": r["numbers"], "target": r["target"],
                                   "correct": c, "extracted": ext}) + "\n")
    if perq:
        perq.close()
    n = len(rows); acc = n_correct / n if n else 0.0
    return {"n": n, "full_n": len(data), "capped": len(rows) < len(data),
            "accuracy": round(acc, 4), "extract_rate": round(n_extract / n, 4) if n else 0.0,
            "headroom": round(1 - acc, 4), "ceiling_risk": (1 - acc) < 0.05,
            "gen_seconds": round(gen_s, 1)}


def load_rows(spec, fn, cap):
    from datasets import load_dataset
    hid, cfg, split = spec
    ds = load_dataset(hid, cfg, split=split) if cfg else load_dataset(hid, split=split)
    rows, capped = [], False
    for r in ds:
        got = fn(r)
        if got is None:
            continue
        q, gold, level = got
        if gold is None or not str(gold).strip():
            continue
        rows.append({"q": q, "gold": str(gold), "level": level})
        if cap and len(rows) >= cap:
            capped = True; break
    return rows, capped, len(ds)


def eval_on_llm(llm, tok, names=None, cap=300, max_tokens=2048, perq_prefix=None, lora_request=None,
                include_countdown=True):
    """Run the battery on a loaded vLLM model. Returns {dataset: metrics}.
    lora_request: if given (a LoRARequest), every eval generate uses that adapter."""
    from vllm import SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench.data_math import build_prompt
    import config as C

    ad = adapters()
    names = names or (list(ad) + (["countdown"] if include_countdown else []))
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out = {}
    for name in names:
        if name == "countdown":
            out[name] = _eval_countdown(llm, tok, cap, max_tokens, perq_prefix, lora_request)
            continue
        spec, fn = ad[name]
        rows, capped, full_n = load_rows(spec, fn, cap)
        prompts = [build_prompt(tok, r["q"]) for r in rows]
        lr = ([lora_request] * len(prompts)) if lora_request is not None else None
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lr, use_tqdm=False)
        gen_s = time.perf_counter() - t0
        n_correct = n_extract = 0
        per_level = defaultdict(lambda: [0, 0])
        perq = open(f"{perq_prefix}__{name}_perq.jsonl", "w") if perq_prefix else None
        for r, o in zip(rows, outs):
            res = math_reward(o.outputs[0].text, r["gold"])
            c = int(res["reward"] >= 1.0)
            n_correct += c; n_extract += int(res["reward_info"]["extracted"])
            if r["level"] is not None:
                per_level[r["level"]][0] += c; per_level[r["level"]][1] += 1
            if perq:
                op = o.outputs[0]
                perq.write(json.dumps({"gold": r["gold"], "correct": c,
                    "extracted": res["reward_info"]["extracted"], "level": r["level"],
                    "resp_len": len(op.token_ids), "truncated": int(op.finish_reason == "length")}) + "\n")
        if perq:
            perq.close()
        n = len(rows); acc = n_correct / n if n else 0.0
        e = {"n": n, "full_n": full_n, "capped": capped, "accuracy": round(acc, 4),
             "extract_rate": round(n_extract / n, 4) if n else 0.0,
             "headroom": round(1 - acc, 4), "ceiling_risk": (1 - acc) < 0.05,
             "gen_seconds": round(gen_s, 1)}
        if per_level:
            e["per_level_accuracy"] = {str(k): round(v[0] / v[1], 4) for k, v in sorted(per_level.items())}
            sub = [per_level[l] for l in C.LEVELS if l in per_level]
            if sub:
                cc, nn = sum(x[0] for x in sub), sum(x[1] for x in sub)
                e["primary_L3_5_accuracy"] = round(cc / nn, 4) if nn else 0.0
        out[name] = e
    return out
