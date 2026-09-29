"""Capture kl.py's base-greedy record from the UNTRAINED base model, to a json file.

WHY THIS EXISTS
    kl.drift(D) is defined against a record captured while the ORIGINAL base weights are live:

        D = mean_t ( logP_base(y_t) - logP_theta(y_t) ),  y = the BASE model's greedy completion

    The ES trainer satisfies that by construction -- es_train_axis.py calls capture_base() at
    step 0, before the first update, and drift() after the last (see :206 and :399).

    eval_grpo.py could not: it only ever has the MERGED TRAINED model loaded, so calling
    capture_base() there recorded the trained model's logprobs of its own greedy output and
    drift() then compared that model to itself. The result is a self-consistency floor near
    zero regardless of how far GRPO actually moved -- which is what the 2026-08-13 run reported
    as kl_proxy_drift=-2.11e-7 and read as "weights barely moved", while the same run was
    2.18 sigma down on ID. That reading was an artifact of this bug, not a measurement.

    Two vLLM engines cannot share a process safely, so the base capture runs HERE, as its own
    process, and hands the record over as json. Run this against the base model, then pass the
    file to eval_grpo.py --kl_base_rec.

The record is model- and prompt-set-specific: it pins C.MODEL, C.KL_N_PROMPTS, C.DATA_SEED,
C.LEVELS, C.TRAIN_SIZE and max_tokens, and eval_grpo.py refuses a record whose model disagrees.
Reusable across every arm of the same model -- capture once, compare many.

Usage: kl_capture.py OUT_JSON [--gpu 0] [--max_tokens N] [--model_dir DIR]
"""
from __future__ import annotations
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="default config.MAX_TOKENS, which is what the ES arms capture with")
    ap.add_argument("--model_dir", default=None,
                    help="override the model to capture from; defaults to config.MODEL. Use this "
                         "ONLY to point at an identical copy of the base weights (e.g. a raw-"
                         "template copy) -- never at a trained model, which is the bug this "
                         "script exists to prevent.")
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    a = ap.parse_args()

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
    from vllm import LLM
    import kl as klmod
    import config as C

    model = a.model_dir or C.MODEL
    max_tokens = a.max_tokens if a.max_tokens is not None else C.MAX_TOKENS
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype=C.DTYPE, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=4096, enforce_eager=False, enable_prefix_caching=False,
              disable_log_stats=True)

    prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
    rec = klmod.capture_base(llm, prompts, max_tokens=max_tokens)

    # Self-drift under the SAME weights: the numerical floor of this estimator. Any later
    # drift() value must be read against this, not against 0 -- and a trained-model number of
    # the same magnitude means the comparison was mis-wired (that is the -2.11e-7 signature).
    self_drift = klmod.drift(llm, tok, rec)
    print(f"[kl_capture] self-drift floor (same weights) = {self_drift:.3e}", flush=True)

    out = {"model": model, "max_tokens": max_tokens, "n_prompts": len(rec),
           "kl_n_prompts": C.KL_N_PROMPTS, "data_seed": C.DATA_SEED, "levels": list(C.LEVELS),
           "train_size": C.TRAIN_SIZE, "dtype": C.DTYPE, "self_drift_floor": self_drift,
           "rec": rec}
    with open(a.out_json, "w") as f:
        json.dump(out, f)
    n_tok = sum(len(r["comp_ids"]) for r in rec)
    print(f"[kl_capture] wrote {a.out_json}: {len(rec)} prompts, {n_tok} tokens", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
