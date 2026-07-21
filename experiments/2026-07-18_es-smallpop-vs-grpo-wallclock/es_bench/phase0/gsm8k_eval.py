#!/usr/bin/env python
"""Phase 0 quick base-accuracy probe on GSM8K using vLLM (greedy pass@1).

Runs one model on one GPU. Doubles as the vLLM smoke test when --n is small.
Reports accuracy, boxed-format rate, extraction rate, avg response length, and
writes a per-question JSONL + a summary JSON.

Usage:
  python gsm8k_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --n 500 --gpu 0 \
      --out_prefix es_bench/phase0/results/gsm8k_qwen2.5-1.5b-instruct
"""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--out_prefix", required=True)
    return p.parse_args()


GT_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def norm_num(s: str):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gt_answer(ans_field: str):
    m = GT_RE.search(ans_field)
    return norm_num(m.group(1)) if m else None


def extract_pred(text: str):
    """Return (value, used_boxed, success)."""
    boxed = BOXED_RE.findall(text)
    if boxed:
        v = norm_num(boxed[-1])
        if v is not None:
            return v, True, True
    nums = NUM_RE.findall(text)
    if nums:
        v = norm_num(nums[-1])
        if v is not None:
            return v, False, True
    return None, False, False


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # ~/.cache/huggingface is root-owned here; keep the datasets cache writable.
    os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.n, len(ds))
    ds = ds.select(range(n))

    tok = AutoTokenizer.from_pretrained(args.model)
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts = []
    for ex in ds:
        msgs = [{"role": "user", "content": ex["question"] + "\n\n" + instruction}]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

    t_load0 = time.perf_counter()
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=False, disable_log_stats=True)
    load_s = time.perf_counter() - t_load0

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t_gen0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    gen_s = time.perf_counter() - t_gen0

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_prefix + "_perq.jsonl"
    n_correct = n_boxed = n_extract = 0
    tot_out_tokens = tot_words = 0
    with open(jsonl_path, "w") as f:
        for ex, o in zip(ds, outs):
            text = o.outputs[0].text
            gt = gt_answer(ex["answer"])
            pred, used_boxed, ok = extract_pred(text)
            correct = int(ok and gt is not None and abs(pred - gt) < 1e-4)
            n_correct += correct
            n_boxed += int(used_boxed)
            n_extract += int(ok)
            n_tok = len(o.outputs[0].token_ids)
            tot_out_tokens += n_tok
            tot_words += len(text.split())
            f.write(json.dumps({
                "question": ex["question"][:200], "gt": gt, "pred": pred,
                "used_boxed": used_boxed, "extracted": ok, "correct": correct,
                "out_tokens": n_tok,
            }) + "\n")

    summary = {
        "model": args.model, "n": n, "gpu": args.gpu,
        "accuracy": n_correct / n,
        "boxed_format_rate": n_boxed / n,
        "extraction_rate": n_extract / n,
        "avg_out_tokens": tot_out_tokens / n,
        "avg_words": tot_words / n,
        "load_seconds": round(load_s, 2),
        "gen_seconds": round(gen_s, 2),
        "gen_tokens_per_s": round(tot_out_tokens / gen_s, 1),
        "max_tokens": args.max_tokens,
    }
    with open(args.out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
