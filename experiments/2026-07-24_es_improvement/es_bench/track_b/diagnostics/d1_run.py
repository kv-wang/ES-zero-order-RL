#!/usr/bin/env python
"""Phase 1 D1: uncapped MATH500-L3-5 eval (per-q correct/len/truncation by difficulty) for any HF
model, + optional GRPO memorization test (eval the checkpoint on its OWN training problems, via
the EVAL pipeline prompt vs the TRAIN pipeline prompt, to separate memorization from template drift)."""
import argparse, json, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grpo_train", type=int, default=0)   # N train problems to re-score (0=skip)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
    os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # try folder
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import eval_core
    from es_bench.shared_reward import math_reward
    from es_bench.data_math import build_prompt

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="float16", gpu_memory_utilization=0.6, max_model_len=4096,
              enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {"tag": a.tag, "model": a.model}

    # --- Task A: uncapped MATH500, per-q len/truncation by level ---
    ev = eval_core.eval_on_llm(llm, tok, names=["math500"], cap=500, max_tokens=2048,
                               perq_prefix=a.out + "_m500", include_countdown=False)
    rows = [json.loads(l) for l in open(a.out + "_m500__math500_perq.jsonl")]
    by = {}
    for r in rows:
        lv = r["level"]; d = by.setdefault(lv, {"n": 0, "c": 0, "len": [], "tr": 0})
        d["n"] += 1; d["c"] += r["correct"]; d["len"].append(r["resp_len"]); d["tr"] += r["truncated"]
    L35 = [r for r in rows if r["level"] in (3, 4, 5)]
    res["math500"] = {"overall_acc": ev["math500"]["accuracy"], "n_total": len(rows),
                      "L3_5_acc": round(np.mean([r["correct"] for r in L35]), 4), "L3_5_n": len(L35),
                      "per_level": {str(k): {"n": v["n"], "acc": round(v["c"]/v["n"], 4),
                                    "mean_len": round(float(np.mean(v["len"])), 1),
                                    "trunc_rate": round(v["tr"]/v["n"], 4)} for k, v in sorted(by.items())}}

    # --- Task B: GRPO memorization (eval-prompt vs train-prompt on own train set) ---
    if a.grpo_train > 0:
        import pandas as pd
        df = pd.read_parquet(os.path.join(HERE, "..", "grpo_ood", "data_math", "train.parquet")).iloc[:a.grpo_train]
        qs = [list(r["prompt"])[0]["content"] for _, r in df.iterrows()]
        gts = [r["reward_model"]["ground_truth"] for _, r in df.iterrows()]
        sp = SamplingParams(temperature=0.0, max_tokens=2048)
        # eval pipeline prompt (build_prompt: the same chat render used at eval time)
        ep = [build_prompt(tok, q) for q in qs]
        oe = llm.generate(ep, sp, use_tqdm=False)
        acc_eval = float(np.mean([math_reward(o.outputs[0].text, g)["reward"] >= 1.0 for o, g in zip(oe, gts)]))
        # train pipeline prompt (verl parquet chat messages rendered directly)
        tp = [tok.apply_chat_template(list(r["prompt"]), tokenize=False, add_generation_prompt=True)
              for _, r in df.iterrows()]
        ot = llm.generate(tp, sp, use_tqdm=False)
        acc_train = float(np.mean([math_reward(o.outputs[0].text, g)["reward"] >= 1.0 for o, g in zip(ot, gts)]))
        res["grpo_train_eval"] = {"n": a.grpo_train, "acc_eval_prompt": round(acc_eval, 4),
                                  "acc_train_prompt": round(acc_train, 4),
                                  "prompts_identical": ep[0] == tp[0],
                                  "eval_prompt_head": ep[0][:400], "train_prompt_head": tp[0][:400]}

    json.dump(res, open(a.out, "w"), indent=2)
    print("D1_RESULT:", json.dumps({k: v for k, v in res.items() if not k.endswith("_head")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
