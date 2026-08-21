#!/usr/bin/env python
"""Paired comparison of every arm against the untrained base, plus the ES signal diagnostics.

Exists because the same three computations were being retyped inline for each arm as it
finished, which makes the numbers in a report unreproducible. Everything here reads only
finished artifacts -- no GPU, safe to run while a training arm is still going.

Three things, in the order they matter:

  1. Paired McNemar vs base on the primary ID metric (MATH-500 L3-5). Paired, not a
     difference of means, because both arms answer the SAME questions: the pairing removes
     question difficulty as a variance source. Reported as net flips and z.

  2. Seed pooling per variant. A single seed's net is small enough to be noise -- the
     2026-08-14 vanilla arms were -7 and 0, which read very differently apart than pooled.

  3. Split-half Spearman rho on member fitness, the ES signal detector. Each step evaluates
     N members on the SAME B problems (CRN); split those B into halves, score each member on
     each half, and correlate the two rankings. If the ranking is real signal it survives the
     split; if it is shot noise from a small B it does not. rho near 0 means the update
     direction is noise no matter what the axis or the model is.

Usage:  ./analyze_arms.py                  # all arms found, base-suffixed dirs
        ./analyze_arms.py --suffix ''      # the -Instruct results instead
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys


def spearman(a, b):
    """Rank correlation with midrank tie handling. Ties dominate here (fitness takes ~3
    distinct values), so midranks are the whole point -- naive ranking would invent order."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mid
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def mcnemar(base_rows, arm_rows, levels):
    pairs = [(x, y) for x, y in zip(base_rows, arm_rows)
             if levels is None or x.get("level") in levels]
    lost = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
    won = sum(1 for x, y in pairs if not x["correct"] and y["correct"])
    n = lost + won
    return lost, won, ((won - lost) / math.sqrt(n) if n else 0.0), len(pairs)


def two_sided_p(z):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def signal_diag(trace_path):
    """rho and fitness resolution from a training trace. `correct_bits` is one bitmask per
    member over the B problems, so it reconstructs per-problem outcomes without rerunning."""
    rows = jsonl(trace_path)
    rhos, distinct, allsame, zero = [], [], 0, 0
    for r in rows:
        cb, B = r.get("correct_bits"), r.get("batch_size")
        if not cb or not B:
            continue
        bits = [[(m >> i) & 1 for i in range(B)] for m in cb]
        h = B // 2
        rhos.append(spearman([sum(b[:h]) for b in bits], [sum(b[h:]) for b in bits]))
        tot = [sum(b) for b in bits]
        distinct.append(len(set(tot)))
        allsame += int(len(set(tot)) == 1)
        zero += int(bool(r.get("zero_update")))
    if not rhos:
        return None
    rhos.sort(); distinct.sort()
    return {
        "steps": len(rhos),
        "rho_med": rhos[len(rhos) // 2],
        "rho_mean": sum(rhos) / len(rhos),
        "rho_over_gate": sum(x > 0.1 for x in rhos),
        "distinct_med": distinct[len(distinct) // 2],
        "B": B,
        "all_tied": allsame,
        "zero_update": zero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=None,
                    help="results dir suffix; defaults to config.RESULTS_SUFFIX")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--levels", default="3,4,5", help="empty string = all levels")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "src"))
    import config as C
    suf = C.RESULTS_SUFFIX if a.suffix is None else a.suffix
    levels = set(int(x) for x in a.levels.split(",")) if a.levels else None
    R = os.path.join(here, "results")

    base_p = os.path.join(R, f"base{suf}", f"base_finaleval__{a.dataset}_perq.jsonl")
    if not os.path.exists(base_p):
        sys.exit(f"no base reference at {base_p} -- run stage 0 first")
    base = jsonl(base_p)

    arms = []
    for p in sorted(glob.glob(os.path.join(R, f"grpo{suf}", f"*_finaleval__{a.dataset}_perq.jsonl"))):
        tag = os.path.basename(p).split("_finaleval__")[0]
        arms.append(("grpo", tag, p, None))
    for v in ("vanilla", "baseaxis", "momentum"):
        for p in sorted(glob.glob(os.path.join(R, f"momentum{suf}",
                                               f"{v}_N*_finaleval__{a.dataset}_perq.jsonl"))):
            tag = os.path.basename(p).split("_finaleval__")[0]
            arms.append((v, tag, p, os.path.join(R, f"momentum{suf}", tag + ".jsonl")))

    if not arms:
        sys.exit(f"no arms found under {R}/*{suf}/")

    print(f"model={C.MODEL}  dataset={a.dataset}  levels={sorted(levels) if levels else 'all'}")
    print(f"base reference: {sum(r['correct'] for r in base)}/{len(base)} overall\n")

    print(f"{'arm':26s} {'base→错':>7s} {'base→对':>7s} {'净':>5s} {'z':>7s} {'p':>6s}   "
          f"{'ρ中位':>7s} {'取值数':>6s} {'0更新':>6s}")
    pooled = {}
    for variant, tag, perq_p, trace_p in arms:
        rows = jsonl(perq_p)
        if len(rows) != len(base) or any(x["gold"] != y["gold"] for x, y in zip(base, rows)):
            print(f"{tag:26s}  !! not aligned with base -- skipped")
            continue
        lost, won, z, n = mcnemar(base, rows, levels)
        d = signal_diag(trace_p) if trace_p and os.path.exists(trace_p) else None
        pooled.setdefault(variant, [0, 0])
        pooled[variant][0] += lost
        pooled[variant][1] += won
        sig = (f"{d['rho_med']:+7.3f} {d['distinct_med']:6d} {d['zero_update']:6d}"
               if d else " " * 21)
        print(f"{tag:26s} {lost:7d} {won:7d} {won-lost:+5d} {z:+7.2f} {two_sided_p(z):6.2f}   {sig}")

    print(f"\n按方法合并 seed(n = 翻转对数):")
    for variant, (lost, won) in pooled.items():
        n = lost + won
        z = (won - lost) / math.sqrt(n) if n else 0.0
        print(f"  {variant:12s} 净 {won-lost:+4d}  z={z:+.2f}  p={two_sided_p(z):.2f}  (n={n})")

    k = len(arms)
    print(f"\n阈值:单项 0.05 → |z|>1.96;{k} 重 Bonferroni → |z|>"
          f"{abs(_probit(0.05 / (2 * k))):.2f}")
    print("ρ 参照:track_b 信号闸门 0.1;07-30 countdown(ES 学得动)0.51")


def _probit(p):
    """Inverse normal CDF, Acklam's rational approximation. Only used to print the
    Bonferroni threshold for however many arms happen to exist, so full precision is
    unnecessary; accuracy is ~1e-9 which is far beyond what a printed threshold needs."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


if __name__ == "__main__":
    main()
