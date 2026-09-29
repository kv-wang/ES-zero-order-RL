#!/usr/bin/env python
"""Offline split-half Spearman rho curve from an ES run's per-step `correct_bits`.

WHY OFFLINE, NOT IN THE TRAINER
-------------------------------
es_train_axis.py already logs everything rho needs: each step's row carries
`correct_bits` (one Python int per member, bit b = that member solved problem b of
this step's CRN batch) and `batch_size`. Computing rho here instead of inside the
trainer means the trainer is not edited, so a run whose vanilla arm has already
finished and whose baseaxis arm has not yet launched keeps byte-identical code
across both arms. That code parity is the whole point -- an arm-vs-arm comparison
where the two arms ran different source is not a comparison.

WHAT RHO MEASURES
-----------------
Whether the fitness ordering of the N population members is a property of the
POLICY or of the particular B problems drawn this step. Permute the B problems into
two disjoint halves, score every member on each half, Spearman-correlate the two
length-N vectors. rho ~ 1: both halves rank the members the same way, so the ES
gradient is reading real signal. rho ~ 0: the ranking is batch noise and the update
direction is arbitrary. Averaged over several independent permutations because a
single split is itself noisy.

This tolerates a truncated final line, so it can be run against a live run's jsonl.

Usage:
    rho_curve.py --jsonl results/.../vanilla_N30_B100_s0.jsonl
    rho_curve.py --jsonl <f> --splits 20 --out <f>_rho.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, random


def popcount(x: int) -> int:
    # int.bit_count() is 3.10+; bin().count is portable and fast enough at B<=1024.
    return bin(x).count("1")


def spearman(x, y):
    """Spearman rho with average ranks for ties. Returns None if either side is constant.

    Rolled by hand rather than pulled from scipy so this script has no dependency
    beyond the stdlib -- it needs to run on a login shell while the GPUs are busy.
    """
    n = len(x)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:      # a half where every member solved the same count
        return None
    return sxy / math.sqrt(sxx * syy)


def step_rho(bits, B, splits, seed0):
    """Mean split-half rho over `splits` independent permutations of the B problems.

    Returns (mean_rho_or_None, n_valid_splits, n_degenerate_splits). A split is
    degenerate when one half gives every member the same solved-count -- that is a
    real and informative state (it is what B=8 produced on the math arms), so it is
    counted separately rather than silently folded in as rho=0.
    """
    vals, degen = [], 0
    for s in range(splits):
        rng = random.Random(seed0 + s)
        idx = list(range(B))
        rng.shuffle(idx)
        half = B // 2
        mask_a = 0
        for i in idx[:half]:
            mask_a |= (1 << i)
        mask_b = 0
        for i in idx[half:2 * half]:      # drop the odd problem so halves are equal size
            mask_b |= (1 << i)
        xa = [popcount(m & mask_a) for m in bits]
        xb = [popcount(m & mask_b) for m in bits]
        r = spearman(xa, xb)
        if r is None:
            degen += 1
        else:
            vals.append(r)
    mean = (sum(vals) / len(vals)) if vals else None
    return mean, len(vals), degen


def spearman_brown(r):
    """Half-length rho -> full-batch reliability. rho is measured on B/2 problems per
    half; the quantity the ES update actually uses is fitness over all B, which is
    more reliable. 2r/(1+r) is the standard step-up. Reported alongside, never
    instead of, the raw half-split number."""
    if r is None or r <= -1.0:
        return None
    return 2.0 * r / (1.0 + r)


def load_rows(path):
    """Reads the trainer's jsonl, skipping a partially-written final line."""
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1          # only ever the last line, mid-flush
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="es_train_axis.py per-step log")
    ap.add_argument("--splits", type=int, default=20, help="permutations averaged per step")
    ap.add_argument("--split_seed", type=int, default=20260823)
    ap.add_argument("--out", default=None, help="default <jsonl minus .jsonl>_rho.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.jsonl):
        print(f"MISSING {a.jsonl}", file=sys.stderr)
        return 1
    rows, bad = load_rows(a.jsonl)
    if not rows:
        print(f"EMPTY {a.jsonl}", file=sys.stderr)
        return 1

    per_step, finite = [], []
    for r in rows:
        bits, B = r.get("correct_bits"), r.get("batch_size")
        if not bits or not B:
            continue
        rho, nv, nd = step_rho(bits, B, a.splits, a.split_seed)
        rec = {
            "step": r["step"],
            "rho_halfsplit": None if rho is None else round(rho, 4),
            "rho_full_sb": None if rho is None else round(spearman_brown(rho), 4),
            "valid_splits": nv, "degenerate_splits": nd,
            "batch_size": B, "population_size": len(bits),
            "fitness_mean": round(sum(r["fitness"]) / len(r["fitness"]), 4) if r.get("fitness") else None,
            "fitness_std": round(r.get("fitness_std", 0.0), 5),
            "solve_rate": round(sum(popcount(m) for m in bits) / (len(bits) * B), 4),
        }
        per_step.append(rec)
        if rho is not None:
            finite.append(rho)

    overall = {
        "jsonl": os.path.abspath(a.jsonl),
        "steps_scored": len(per_step),
        "truncated_lines_skipped": bad,
        "splits_per_step": a.splits,
        "split_seed": a.split_seed,
        "rho_mean": round(sum(finite) / len(finite), 4) if finite else None,
        "rho_min": round(min(finite), 4) if finite else None,
        "rho_max": round(max(finite), 4) if finite else None,
        "steps_all_degenerate": sum(1 for p in per_step if p["rho_halfsplit"] is None),
        "per_step": per_step,
    }
    out = a.out or (a.jsonl[:-6] if a.jsonl.endswith(".jsonl") else a.jsonl) + "_rho.json"
    with open(out, "w") as f:
        json.dump(overall, f, indent=2)

    if not a.quiet:
        print(f"== split-half Spearman rho :: {os.path.basename(a.jsonl)} ==")
        print(f"   steps={len(per_step)}  splits/step={a.splits}"
              + (f"  (skipped {bad} truncated line(s))" if bad else ""))
        print(f"{'step':>5} {'rho_half':>9} {'rho_full':>9} {'solve':>7} {'f_mean':>8} {'f_std':>8} {'degen':>6}")
        for p in per_step:
            rh = "  n/a" if p["rho_halfsplit"] is None else f"{p['rho_halfsplit']:+.3f}"
            rf = "  n/a" if p["rho_full_sb"] is None else f"{p['rho_full_sb']:+.3f}"
            print(f"{p['step']:>5} {rh:>9} {rf:>9} {p['solve_rate']:>7.3f} "
                  f"{(p['fitness_mean'] if p['fitness_mean'] is not None else 0):>8.4f} "
                  f"{p['fitness_std']:>8.5f} {p['degenerate_splits']:>6}")
        print(f"   rho_mean={overall['rho_mean']}  min={overall['rho_min']}  max={overall['rho_max']}")
        print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
