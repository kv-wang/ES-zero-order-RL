#!/usr/bin/env python
"""Merge a verl FSDP checkpoint to HF and score it on the SAME battery as the ES arms.

The output is written in the ES arms' `*_summary.json` schema (`eval_final`, `kl_proxy_drift`,
`wall_clock_s`, ...), so axis_probe/compare_all.py and the run scripts' readout tables can put
GRPO rows next to momentum / baseaxis / vanilla / sigadapt rows without special-casing.

Identical to the ES arms: eval_core.eval_on_llm, cap 300, max_tokens=training cap,
the six pinned sets,
the post-fix brace-counting extractor, fp16, greedy. Per-question jsonl is written with the same
`__<dataset>_perq.jsonl` suffix so paired McNemar against any ES arm works out of the box.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor_dir", required=True, help="verl .../global_step_N/actor")
    ap.add_argument("--out_prefix", required=True, help="mirrors the ES arms' --out_prefix")
    ap.add_argument("--include_countdown", action="store_true",
                    help="add countdown (the ID metric for a countdown-trained arm) to the battery")
    ap.add_argument("--restore_tokenizer", default=None,
                    help="tokenizer_config.orig.json from make_raw_template_model.py; a countdown "
                         "run trains on the identity-template copy and must restore the real one")
    ap.add_argument("--eval_cap", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="completion cap; default config.MAX_TOKENS (same as training)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--kl", action="store_true", help="also report the KL-to-base proxy")
    ap.add_argument("--train_seconds", type=float, default=None,
                    help="verl training wall clock, carried into the summary for budget matching")
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--rollouts_per_step", type=int, default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    HERE = os.path.dirname(os.path.abspath(__file__))
    TRY = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    sys.path.insert(0, TRY)
    sys.path.insert(0, os.path.join(HERE, "..", "src"))

    hf_dir = os.path.join(os.path.dirname(a.actor_dir), "hf_merged")
    if not os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"[merge] {a.actor_dir} -> {hf_dir}", flush=True)
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", a.actor_dir, "--target_dir", hf_dir], check=True)

    # A countdown GRPO run trains against the IDENTITY-chat-template model copy so verl's
    # rollout string matches the ES arms' raw `context`. The merged dir inherits that identity
    # template -- restore the real one, or every math OOD prompt below is built with a template
    # that strips the chat markers the model was pretrained on.
    if a.restore_tokenizer and os.path.exists(hf_dir):
        shutil.copy(a.restore_tokenizer, os.path.join(hf_dir, "tokenizer_config.json"))
        print(f"[merge] restored real tokenizer_config from {a.restore_tokenizer}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM
    import eval_core
    import config as C

    tok = AutoTokenizer.from_pretrained(hf_dir)
    llm = LLM(model=hf_dir, dtype=C.DTYPE, gpu_memory_utilization=0.85, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)

    kl_drift = None
    if a.kl:
        # Same fixed prompt set and estimator the ES arms use, so the KL column is comparable.
        import kl as klmod
        kl_prompts = klmod.build_kl_prompts(tok, C.KL_N_PROMPTS, C.DATA_SEED, C.LEVELS, C.TRAIN_SIZE)
        rec = klmod.capture_base(llm, kl_prompts, max_tokens=C.MAX_TOKENS)
        kl_drift = klmod.drift(llm, tok, rec)
        print(f"[kl] drift={kl_drift:.4f}", flush=True)

    t0 = time.perf_counter()
    eval_max_tokens = a.max_tokens if a.max_tokens is not None else C.MAX_TOKENS
    ev = eval_core.eval_on_llm(llm, tok, cap=a.eval_cap, max_tokens=eval_max_tokens,
                               perq_prefix=a.out_prefix + "_finaleval",
                               include_countdown=a.include_countdown)
    eval_seconds = time.perf_counter() - t0

    gens = (a.train_steps * a.rollouts_per_step) if (a.train_steps and a.rollouts_per_step) else None
    summary = {
        "variant": "grpo", "axis": None, "model": C.MODEL,
        "hf_dir": hf_dir, "actor_dir": a.actor_dir,
        "num_steps": a.train_steps, "rollouts_per_step": a.rollouts_per_step,
        "total_generations": gens,
        "wall_clock_s": a.train_seconds,
        "kl_proxy_drift": kl_drift,
        "eval_max_tokens": eval_max_tokens,
        "eval_final": ev, "eval_seconds": eval_seconds,
        # ES-only fields, present as None so downstream tables line up
        "population_size": None, "pop_seed": None, "pair_tie_rate": None,
        "final_cum_disp": None, "zero_update_rate": None, "s_per_step_mean": None,
    }
    with open(a.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("EVAL_FINAL:", json.dumps(ev, indent=2), flush=True)


if __name__ == "__main__":
    main()
