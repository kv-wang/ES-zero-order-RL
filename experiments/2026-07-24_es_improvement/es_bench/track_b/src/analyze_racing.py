#!/usr/bin/env python
"""Does a two-stage (racing) fitness estimator buy update fidelity per rollout?

Uniform estimator: every member scored on B' problems.  Cost = N*B' sequences.
Racing estimator:  every member scored on B1 problems, then the top-k and bottom-k by that
                   cheap score are re-scored on all B problems (they carry the large |z| that
                   dominates Delta = sum_i z_i eps_i); the middle keeps its cheap score.
                   Cost = N*B1 + 2k*(B-B1) sequences.

Fidelity = Pearson r between the resulting z-vector and the full-batch z at B=200, which equals
cos(Delta_est, Delta_full) for near-orthogonal eps.  Wall-clock is modelled from the measured
throughput curve (tok/s = 69.0 * seqs^0.603), so cost_s ~ seqs^0.397 per step.
"""
import argparse, json, math
import numpy as np
from analyze_batch_scaling import decode, pearson


def zs(f):
    return (f - f.mean()) / (f.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl)]
    mats = [decode(r["correct_bitmap_hex"], a.B) for r in rows]
    N = mats[0].shape[0]
    rng = np.random.default_rng(a.seed)
    P, K = 0.603, 69.0
    unit = (N * a.B) / (K * (N * a.B) ** P)          # seconds-equivalent of the full B=200 step

    def sec(seqs):
        return (seqs / (K * seqs ** P)) / unit        # relative to the full step

    print(f"N={N} B={a.B} steps={len(mats)}   (fidelity = corr with full-batch z; "
          f"cost relative to the 194.7 s/step full config)\n")
    print(f"{'estimator':34s} {'seqs':>7s} {'rel_cost':>9s} {'s/step':>8s} {'fidelity':>9s} "
          f"{'fid/sqrt(cost)':>15s}")

    def report(name, seqs, fids):
        c = sec(seqs)
        f = float(np.mean(fids))
        print(f"{name:34s} {seqs:7d} {c:9.3f} {c*194.7:8.1f} {f:9.3f} {f/math.sqrt(c):15.3f}")

    for Bp in [25, 50, 100, 200]:
        fids = []
        for M in mats:
            zf = zs(M.mean(axis=1))
            for _ in range(a.draws if Bp < a.B else 1):
                cols = rng.choice(a.B, Bp, replace=False)
                f = M[:, cols].mean(axis=1)
                fids.append(pearson(zs(f), zf) if f.std() > 1e-12 else 0.0)
        report(f"uniform B={Bp}", N * Bp, fids)

    print()
    for B1 in [25, 50]:
        for k in [4, 8]:
            fids = []
            for M in mats:
                zf = zs(M.mean(axis=1))
                for _ in range(a.draws):
                    cols = rng.choice(a.B, B1, replace=False)
                    f1 = M[:, cols].mean(axis=1)
                    if f1.std() < 1e-12:
                        fids.append(0.0); continue
                    order = np.argsort(f1)
                    sel = np.concatenate([order[:k], order[-k:]])
                    f = f1.copy()
                    f[sel] = M[sel].mean(axis=1)      # survivors get the full-batch score
                    fids.append(pearson(zs(f), zf) if f.std() > 1e-12 else 0.0)
            report(f"racing B1={B1} top/bot k={k}", N * B1 + 2 * k * (a.B - B1), fids)


if __name__ == "__main__":
    main()
