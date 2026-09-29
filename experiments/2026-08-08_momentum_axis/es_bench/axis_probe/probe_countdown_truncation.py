#!/usr/bin/env python
"""Why does countdown extract_rate fall after training? Truncation, or format loss?

eval_core._eval_countdown scores extraction as `"<answer>" in txt and "</answer>" in txt`,
which conflates two very different failures:

  (a) the model emitted a complete response that simply lacks the tags  -> format loss
  (b) the model was still searching inside <think> when it hit max_tokens -> truncation

The per-question JSONL that eval_core writes keeps neither the text nor the token count, so
the distinction is not recoverable from the artefacts on disk. This script re-runs the SAME
eval slice under the SAME decoding settings as eval_grpo.py:73 and additionally records
`n_tokens` and vLLM's `finish_reason` per question, which separates (a) from (b) directly.

It changes nothing about the eval itself -- accuracy and extract_rate printed here should
reproduce the arm's summary.json (vLLM is not bit-deterministic; expect +/-1-2 points).

Usage:
  python probe_countdown_truncation.py <model_or_hf_dir> <out_jsonl> [--gpu 0] [--cap 300]
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTDOWN_JSON = "/home/hyin66/es-fine-tuning-paper/countdown/data/countdown.json"
COUNTDOWN_DIR = "/home/hyin66/es-fine-tuning-paper/countdown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=2048)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    sys.path.insert(0, os.path.join(HERE, "src"))
    sys.path.insert(0, COUNTDOWN_DIR)
    import config as C
    from countdown_task import answer_reward_function
    from vllm import LLM, SamplingParams

    rows = json.load(open(COUNTDOWN_JSON))[:a.cap]
    prompts = [r["context"] for r in rows]

    # Identical to eval_grpo.py:73 and eval_core.eval_on_llm, so the numbers are comparable.
    llm = LLM(model=a.model, dtype=C.DTYPE, gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=C.STOP)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    recs = []
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            g = o.outputs[0]
            txt = g.text
            rec = {
                "numbers": r["numbers"], "target": r["target"],
                "n_numbers": len(r["numbers"]),
                "correct": int(answer_reward_function(txt, r["numbers"], r["target"]) >= 1.0),
                "extracted": int("<answer>" in txt and "</answer>" in txt),
                "has_open_answer": int("<answer>" in txt),
                "has_close_think": int("</think>" in txt),
                "n_tokens": len(g.token_ids),
                "finish_reason": g.finish_reason,
                "stop_reason": str(g.stop_reason),
            }
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")

    n = len(recs)
    ext = [r for r in recs if r["extracted"]]
    noext = [r for r in recs if not r["extracted"]]
    trunc = [r for r in noext if r["finish_reason"] == "length"]
    print(f"\n=== {a.model}")
    print(f"n={n}  accuracy={sum(r['correct'] for r in recs)/n:.4f}  "
          f"extract_rate={len(ext)/n:.4f}")
    print(f"no-tag questions: {len(noext)}")
    print(f"  of which finish_reason=='length' (hit the {a.max_tokens}-token cap): "
          f"{len(trunc)}  ({len(trunc)/len(noext) if noext else 0:.3f})")
    print(f"  of which stopped normally but emitted no tag: {len(noext)-len(trunc)}")
    print(f"  opened <answer> but never closed it: {sum(r['has_open_answer'] for r in noext)}")
    print(f"  never closed </think> either: {len(noext)-sum(r['has_close_think'] for r in noext)}")
    if ext:
        m = sorted(r["n_tokens"] for r in ext)
        print(f"median n_tokens | tag closed : {m[len(m)//2]}")
    if noext:
        m = sorted(r["n_tokens"] for r in noext)
        print(f"median n_tokens | no tag     : {m[len(m)//2]}")
    for k in (3, 4):
        sub = [r for r in recs if r["n_numbers"] == k]
        if sub:
            print(f"  {k}-number: n={len(sub)} extract={sum(r['extracted'] for r in sub)/len(sub):.4f} "
                  f"acc={sum(r['correct'] for r in sub)/len(sub):.4f} "
                  f"trunc_rate={sum(r['finish_reason']=='length' for r in sub)/len(sub):.4f}")


if __name__ == "__main__":
    main()
