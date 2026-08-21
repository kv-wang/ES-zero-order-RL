#!/usr/bin/env python
"""Phase 4 accuracy eval battery (greedy pass@1, house extraction + reward).

Evaluates ONE model (a HF id or a local checkpoint dir) on the pinned datasets.
Per dataset: accuracy, n, extraction-success rate; math500 also per-level and the
L3-5 primary-ID subset. Writes per-q JSONL + a summary JSON; flags <5pt headroom.

Usage (from es_bench dir, after `source phase4_continuous_ood/env.sh`):
  CUDA_VISIBLE_DEVICES=0 $P4_PY phase4_continuous_ood/eval/eval_model.py \
      --model Qwen/Qwen2.5-Math-1.5B-Instruct --tag base --cap 500
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder -> es_bench importable
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def adapters():
    """name -> (loader_kwargs, row->(-question, gold, level) or None to skip)."""
    import config as C

    def math500(r):
        return r["problem"], r["answer"], int(r["level"])

    def gsm8k(r):
        from es_bench.data_math import _gsm_gt
        return r["question"], _gsm_gt(r["answer"]), None

    def svamp(r):
        return (r["Body"].rstrip() + " " + r["Question"]).strip(), str(r["Answer"]), None

    def minerva(r):
        return r["question"], str(r["answer"]), None

    def olympiad(r):
        if any(r.get(f"image_{i}") for i in range(1, 10)):
            return None  # skip image problems (text-only battery)
        fa = r["final_answer"]
        gold = (fa[0] if isinstance(fa, list) and fa else fa)
        if gold is None:
            return None
        return r["question"], str(gold).strip(), None

    def amc23(r):
        return r["question"], str(r["answer"]), None

    return {
        "math500": (C.EVAL_DATASETS["math500"], math500),
        "gsm8k": (C.EVAL_DATASETS["gsm8k"], gsm8k),
        "svamp": (C.EVAL_DATASETS["svamp"], svamp),
        "minerva_math": (C.EVAL_DATASETS["minerva_math"], minerva),
        "olympiadbench": (C.EVAL_DATASETS["olympiadbench"], olympiad),
        "amc23": (C.EVAL_DATASETS["amc23"], amc23),
    }


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
            capped = True
            break
    return rows, capped, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="e.g. base / N4_cont_s0_step100")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--cap", type=int, default=500, help="max rows/dataset (0=full)")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "results", "eval"))
    a = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from es_bench.shared_reward import math_reward
    from es_bench.data_math import build_prompt
    import config as C

    ad = adapters()
    names = list(ad) if a.datasets == "all" else a.datasets.split(",")
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype=C.DTYPE, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, enforce_eager=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model, "tag": a.tag, "cap": a.cap, "max_tokens": a.max_tokens, "datasets": {}}

    for name in names:
        spec, fn = ad[name]
        rows, capped, full_n = load_rows(spec, fn, a.cap)
        prompts = [build_prompt(tok, r["q"]) for r in rows]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        gen_s = time.perf_counter() - t0

        n_correct = n_extract = 0
        per_level = defaultdict(lambda: [0, 0])
        perq_path = out_dir / f"{a.tag}__{name}_perq.jsonl"
        with open(perq_path, "w") as f:
            for r, o in zip(rows, outs):
                text = o.outputs[0].text
                res = math_reward(text, r["gold"])
                c = int(res["reward"] >= 1.0)
                n_correct += c
                n_extract += int(res["reward_info"]["extracted"])
                if r["level"] is not None:
                    per_level[r["level"]][0] += c
                    per_level[r["level"]][1] += 1
                f.write(json.dumps({"gold": r["gold"], "correct": c,
                                    "extracted": res["reward_info"]["extracted"],
                                    "level": r["level"],
                                    "out_tokens": len(o.outputs[0].token_ids)}) + "\n")
        n = len(rows)
        acc = n_correct / n if n else 0.0
        entry = {"n": n, "full_n": full_n, "capped": capped,
                 "accuracy": round(acc, 4),
                 "extract_rate": round(n_extract / n, 4) if n else 0.0,
                 "headroom": round(1.0 - acc, 4),
                 "ceiling_risk": (1.0 - acc) < 0.05,
                 "gen_seconds": round(gen_s, 1)}
        if per_level:
            entry["per_level_accuracy"] = {str(k): round(v[0] / v[1], 4) for k, v in sorted(per_level.items())}
            entry["per_level_n"] = {str(k): v[1] for k, v in sorted(per_level.items())}
            sub = [per_level[l] for l in C.LEVELS if l in per_level]
            if sub:
                cc, nn = sum(x[0] for x in sub), sum(x[1] for x in sub)
                entry["primary_L3_5_accuracy"] = round(cc / nn, 4) if nn else 0.0
                entry["primary_L3_5_headroom"] = round(1.0 - cc / nn, 4) if nn else 1.0
        report["datasets"][name] = entry
        print(f"[{name}] n={n}{'(capped from %d)'%full_n if capped else ''} "
              f"acc={acc:.4f} extract={entry['extract_rate']:.3f} "
              f"{'*CEILING RISK*' if entry['ceiling_risk'] else ''}", flush=True)

    summ_path = out_dir / f"{a.tag}__summary.json"
    with open(summ_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nWROTE", summ_path)
    print(json.dumps(report["datasets"], indent=2))


if __name__ == "__main__":
    main()
