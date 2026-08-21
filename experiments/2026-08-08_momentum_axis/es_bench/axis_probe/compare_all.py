#!/usr/bin/env python
"""One table across every arm in this try: momentum / baseaxis / vanilla / sigadapt / grpo.

All arms write the same `*_summary.json` schema and the same `__<dataset>_perq.jsonl` files, so
this reads whatever is on disk. Paired McNemar is computed against a chosen reference arm using
the per-question files -- the marginal accuracies alone cannot separate a real effect from the
~4 questions of noise that separate these arms.

Usage:
    python axis_probe/compare_all.py                    # table + paired tests vs vanilla
    python axis_probe/compare_all.py --ref grpo_gen400  # paired tests vs the GRPO arm
"""
from __future__ import annotations
import argparse, glob, json, math, os

SETS = ["math500", "svamp", "gsm8k", "minerva_math", "olympiadbench", "amc23"]
OOD = ["svamp", "gsm8k", "minerva_math", "olympiadbench"]


def load(root):
    arms = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*_summary.json"))):
        tag = os.path.basename(f).replace("_summary.json", "")
        d = json.load(open(f))
        if "eval_final" not in d:
            continue
        d["_dir"] = os.path.dirname(f)
        d["_tag"] = tag
        arms[tag] = d
    return arms


def perq(d, dataset):
    p = os.path.join(d["_dir"], f"{d['_tag']}_finaleval__{dataset}_perq.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def mcnemar(a_rows, b_rows, level_filter=None):
    """b = arm under test, c = reference. Returns (b, c, z) or None if unpairable."""
    if a_rows is None or b_rows is None or len(a_rows) != len(b_rows):
        return None
    if any(x["gold"] != y["gold"] for x, y in zip(a_rows, b_rows)):
        return None                      # different question order -> not pairable
    pair = list(zip(a_rows, b_rows))
    if level_filter:
        pair = [(x, y) for x, y in pair if x.get("level") in level_filter]
    b = sum(1 for x, y in pair if x["correct"] and not y["correct"])
    c = sum(1 for x, y in pair if y["correct"] and not x["correct"])
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    return b, c, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--ref", default=None, help="reference arm for paired tests (default: a vanilla arm)")
    a = ap.parse_args()

    arms = load(a.root)
    if not arms:
        print(f"no summaries under {a.root}"); return

    def g(d, *path, default=None):
        for k in path:
            d = (d or {}).get(k)
        return d if d is not None else default

    print("=== all arms ===")
    hdr = f"{'arm':<22}{'ID(L3-5)':>10}{'OOD-avg':>9}{'KLx1e3':>8}{'gens':>9}{'train_s':>9}"
    print(hdr)
    for tag, d in arms.items():
        ood = [g(d, "eval_final", s, "accuracy") for s in OOD]
        ood = [x for x in ood if x is not None]
        idv = g(d, "eval_final", "math500", "primary_L3_5_accuracy")
        kl = d.get("kl_proxy_drift")
        gens = d.get("total_generations")
        if gens is None and d.get("population_size"):
            gens = d["num_steps"] * d["population_size"] * 8
        wc = d.get("wall_clock_s")
        c_id = f"{idv:.4f}" if idv is not None else ""
        c_ood = f"{sum(ood)/len(ood):.4f}" if len(ood) == 4 else ""
        c_kl = f"{kl*1e3:.3f}" if kl is not None else ""
        c_gen = str(gens) if gens else ""
        c_wc = f"{wc:.0f}" if wc else ""
        print(f"{tag:<22}{c_id:>10}{c_ood:>9}{c_kl:>8}{c_gen:>9}{c_wc:>9}")

    ref_tag = a.ref or next((t for t in arms if t.startswith("vanilla")), None)
    if ref_tag is None or ref_tag not in arms:
        print("\n(no reference arm for paired tests)"); return
    ref = arms[ref_tag]

    print(f"\n=== paired McNemar vs {ref_tag}  (z>0 means the arm beats the reference) ===")
    print(f"{'arm':<22}" + "".join(f"{s[:9]:>11}" for s in SETS))
    for tag, d in arms.items():
        if tag == ref_tag:
            continue
        cells = []
        for s in SETS:
            r = mcnemar(perq(d, s), perq(ref, s), level_filter=(3, 4, 5) if s == "math500" else None)
            cells.append("n/a" if r is None else f"{r[2]:+.2f}")
        print(f"{tag:<22}" + "".join(f"{c:>11}" for c in cells))
    print("\nSix tests per row: Bonferroni needs |z| >= 2.64, not 1.96.")
    print("amc23 is n=40 per seed and pre-flagged high-variance in config.py; it has produced the")
    print("largest |z| in this repo twice, with opposite signs. Treat it as noise unless replicated.")
    print("Budget is NOT equalised across methods by construction -- read `gens` and `train_s`")
    print("together before reading accuracy.")


if __name__ == "__main__":
    main()
