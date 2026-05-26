from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_extraction import exact_match, extract_final_answer
from datasets import build_source_splits, load_eval_dataset


def ensure_dirs():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("artifacts/generations").mkdir(parents=True, exist_ok=True)


def generate(model, tok, prompt, max_new_tokens=256, temperature=0.0):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def eval_dataset(model, tok, examples, k=1, temperature=0.0):
    scores, format_ok, extract_ok, lengths = [], [], [], []
    for ex in examples:
        candidate_scores = []
        extracted_any = False
        boxed_any = False
        lens = []
        for _ in range(k):
            output = generate(model, tok, ex.question, temperature=temperature)
            acc, ext = exact_match(output, ex.answer)
            candidate_scores.append(acc)
            extracted_any = extracted_any or ext.success
            boxed_any = boxed_any or ext.used_boxed
            lens.append(len(output.split()))
        scores.append(max(candidate_scores))
        extract_ok.append(int(extracted_any))
        format_ok.append(int(boxed_any))
        lengths.append(mean(lens))
    return {
        "pass": mean(scores) if scores else 0.0,
        "format_success_rate": mean(format_ok) if format_ok else 0.0,
        "extraction_success_rate": mean(extract_ok) if extract_ok else 0.0,
        "avg_response_length": mean(lengths) if lengths else 0.0,
    }


def fake_es_update(model, sigma, alpha):
    # Lightweight placeholder ES update step for pipeline wiring.
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * sigma
            p.add_(alpha * noise)


def run_train(args, model, tok):
    train, val = build_source_splits(args.train_dataset, args.train_size, args.val_size, args.seed)
    rows = []
    total_evals = 0
    start = time.time()

    for step in range(1, args.num_steps + 1):
        fake_es_update(model, args.sigma, args.alpha)
        train_eval = eval_dataset(model, tok, train[: args.eval_subset], k=1, temperature=0.0)
        val_eval = eval_dataset(model, tok, val[: args.eval_subset], k=1, temperature=0.0)
        total_evals += args.population_size * args.eval_subset

        rows.append(
            {
                "method": args.method,
                "train_dataset": args.train_dataset,
                "step": step,
                "train_reward": train_eval["pass"],
                "val_reward": val_eval["pass"],
                "population_size": args.population_size,
                "sigma": args.sigma,
                "alpha": args.alpha,
            }
        )

        if step % args.save_every == 0 or step == args.num_steps:
            ckpt = f"artifacts/checkpoints/{args.method}_{args.train_dataset}_step{step}.pt"
            torch.save(model.state_dict(), ckpt)

    with open("results/training_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "total_forward_evals": total_evals,
        "wall_clock_seconds": time.time() - start,
    }
    Path("artifacts/train_meta.json").write_text(json.dumps(meta, indent=2))


def select_checkpoints(args):
    curves = []
    with open("results/training_curves.csv") as f:
        for r in csv.DictReader(f):
            curves.append(r)

    best = max(curves, key=lambda r: float(r["val_reward"]))
    final = max(curves, key=lambda r: int(r["step"]))
    early_candidates = [r for r in curves if int(r["step"]) < int(best["step"])]
    early = max(early_candidates, key=lambda r: float(r["val_reward"])) if early_candidates else None

    selected = {
        "best": int(best["step"]),
        "final": int(final["step"]),
        "early": int(early["step"]) if early else None,
    }
    Path("artifacts/selected_checkpoints.json").write_text(json.dumps(selected, indent=2))


def run_eval(args, model, tok):
    selected = json.loads(Path("artifacts/selected_checkpoints.json").read_text())
    train_meta = json.loads(Path("artifacts/train_meta.json").read_text())
    out_rows = []

    if args.train_dataset == "math":
        eval_sets = ["math500", "aime2024", "amc", "minerva", "olympiadbench"]
    else:
        eval_sets = ["svamp", "asdiv", "gsm-hard", "math500"]

    for sel_name, step in selected.items():
        if step is None:
            continue
        ckpt = f"artifacts/checkpoints/{args.method}_{args.train_dataset}_step{step}.pt"
        state = torch.load(ckpt, map_location=model.device)
        model.load_state_dict(state)

        for ds_name in eval_sets:
            data = load_eval_dataset(ds_name)[: args.eval_size]
            pass1 = eval_dataset(model, tok, data, k=1, temperature=0.0)
            pass4 = eval_dataset(model, tok, data, k=4, temperature=args.sample_temperature)
            pass8 = eval_dataset(model, tok, data, k=8, temperature=args.sample_temperature)
            out_rows.append(
                {
                    "train_dataset": args.train_dataset,
                    "eval_dataset": ds_name,
                    "model": args.model_name,
                    "method": args.method,
                    "population_size": args.population_size,
                    "sigma": args.sigma,
                    "alpha": args.alpha,
                    "checkpoint_step": step,
                    "selected_by": sel_name,
                    "pass_at_1": pass1["pass"],
                    "pass_at_4": pass4["pass"],
                    "pass_at_8": pass8["pass"],
                    "format_success_rate": pass1["format_success_rate"],
                    "extraction_success_rate": pass1["extraction_success_rate"],
                    "avg_response_length": pass1["avg_response_length"],
                    "total_forward_evals": train_meta["total_forward_evals"],
                    "wall_clock_seconds": train_meta["wall_clock_seconds"],
                }
            )

    with open("results/ood_eval_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--method", default="es")
    p.add_argument("--train_dataset", choices=["math", "gsm8k"], default="math")
    p.add_argument("--population_size", type=int, default=20)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=5e-4)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--eval_subset", type=int, default=64)
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--sample_temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["train", "select", "eval", "all"], default="all")
    args = p.parse_args()

    ensure_dirs()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.model_name)

    if args.stage in ("train", "all"):
        run_train(args, model, tok)
    if args.stage in ("select", "all"):
        select_checkpoints(args)
    if args.stage in ("eval", "all"):
        run_eval(args, model, tok)


if __name__ == "__main__":
    main()
