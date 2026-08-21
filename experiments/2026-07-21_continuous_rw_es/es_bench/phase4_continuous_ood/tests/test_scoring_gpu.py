"""Phase 0c GPU tests (need vLLM + 1 GPU). Run from the es_bench dir:
    CUDA_VISIBLE_DEVICES=0 python phase4_continuous_ood/tests/test_scoring_gpu.py

Validates:
  A. scoring under member weights does NOT corrupt live weights or the fp32 base;
     es_restore_base returns live weights bit-for-bit to base.
  B. the scoring pass is deterministic under fixed weights.
  C. the scoring pass actually reflects the live weights (base vs member differ).
"""
from __future__ import annotations
import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.config")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# es_bench importable = parent of es_bench dir (the try folder)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from es_bench import data_math
from es_bench.phase4_continuous_ood.src import config as C
import scoring


def approx(a, b, rel=1e-9, abs_=1e-6):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def main():
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    train, _ = data_math.make_split(C.DATASET, 64, 8, C.DATA_SEED, levels=C.LEVELS)
    probe = train[:6]
    prompts = [data_math.build_prompt(tok, b["question"]) for b in probe]
    gts = [b["gt"] for b in probe]

    llm = LLM(model=C.MODEL, dtype=C.DTYPE, gpu_memory_utilization=0.85,
              max_model_len=2048, enforce_eager=False, disable_log_stats=True,
              enable_prefix_caching=False,
              worker_extension_cls="es_bench.phase4_continuous_ood.src.phase4_worker.Phase4Worker")
    sp = SamplingParams(temperature=0.0, max_tokens=C.MAX_TOKENS, seed=42)

    def live():  return llm.collective_rpc("es_live_checksum")[0]
    def base():  return llm.collective_rpc("es_base_checksum")[0]

    llm.collective_rpc("es_snapshot_base")
    base0, live0 = base(), live()
    print(f"[snapshot] base sum={base0['sum']:.6f} live sum={live0['sum']:.6f}")

    fails = []
    def check(cond, name):
        print(("  PASS " if cond else "  FAIL ") + name)
        if not cond:
            fails.append(name)

    SEED = 123456
    # --- generate member outputs under member weights ---
    llm.collective_rpc("es_set_member", args=(SEED, C.SIGMA))
    live_m = live()
    check(not approx(live_m["sum"], live0["sum"], abs_=1e-3), "A0 es_set_member changed live weights")
    check(approx(base()["sum"], base0["sum"]) and approx(base()["sumsq"], base0["sumsq"]),
          "A1 base intact after es_set_member")
    member_outs = llm.generate(prompts, sp, use_tqdm=False)
    gen_texts = [o.outputs[0].text for o in member_outs]

    # --- scoring pass under member weights ---
    sec_m, pbar_m = scoring.score_secondary(llm, tok, prompts, gen_texts, gts, C.ANSWER_OPENER)
    print(f"[score@member] secondary={sec_m:.6f} pbar={[round(p,4) for p in pbar_m]}")
    check(all(0.0 <= p <= 1.0 for p in pbar_m) and sec_m > 0.0, "A2 secondary in (0,1], non-degenerate")
    check(approx(live()["sum"], live_m["sum"]) and approx(live()["sumsq"], live_m["sumsq"]),
          "A3 scoring did NOT corrupt live (member) weights")
    check(approx(base()["sum"], base0["sum"]) and approx(base()["sumsq"], base0["sumsq"]),
          "A4 scoring did NOT corrupt fp32 base")

    # --- restore base ---
    llm.collective_rpc("es_restore_base")
    check(approx(live()["sum"], live0["sum"]) and approx(live()["sumsq"], live0["sumsq"]),
          "A5 es_restore_base returned live weights bit-for-bit to base")

    # --- B: determinism under fixed weights (re-set same member, same inputs) ---
    llm.collective_rpc("es_set_member", args=(SEED, C.SIGMA))
    sec_m2, pbar_m2 = scoring.score_secondary(llm, tok, prompts, gen_texts, gts, C.ANSWER_OPENER)
    check(all(approx(a, b) for a, b in zip(pbar_m, pbar_m2)), "B  scoring deterministic under fixed weights")

    # --- C: scoring reflects the live weights (base vs member, same gen/gts) ---
    llm.collective_rpc("es_restore_base")
    sec_b, pbar_b = scoring.score_secondary(llm, tok, prompts, gen_texts, gts, C.ANSWER_OPENER)
    print(f"[score@base]   secondary={sec_b:.6f}")
    check(not approx(sec_b, sec_m, abs_=1e-6), "C  scoring reflects live weights (base != member)")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
