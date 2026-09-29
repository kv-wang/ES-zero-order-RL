#!/usr/bin/env python
"""Offline B-scaling analysis for LoRA-ES — no GPU, no retraining.

`es_lora_main.py` logs `correct_bitmap_hex`: for every step, the N x B binary matrix of
"member i solved problem j". That is the complete fitness evidence at B=200, so any smaller
batch B' < B can be simulated exactly by column-subsampling, and we can measure what cutting
the batch costs the ES update.

Two quantities per (step, B'):

  rho_split   split-half Spearman of the member ranking within the B' subsample -- "is the
              ranking real or is it noise" (the Phase-0 criterion), and

  cos_update  Pearson r between the z-scores at B' and the z-scores at the full B=200.
              The ES update is Delta = (alpha/N) * sum_i z_i eps_i with the eps_i fixed and
              near-orthogonal in high dimension, so cos(Delta_B', Delta_B) = corr(z_B', z_B).
              This is directly the fraction of the full-batch update direction retained.

Figure of merit: a run with per-step cost c(B') and direction fidelity f(B') accumulates useful
displacement at rate f(B')/c(B') per second, so that ratio (normalised to B=200) picks the batch.
"""
import argparse, json, math, os, sys
import numpy as np


def decode(hexes, B):
    """N hex strings -> N x B uint8 matrix (leading zeros were lost by '%x')."""
    out = np.zeros((len(hexes), B), dtype=np.uint8)
    for i, h in enumerate(hexes):
        bits = bin(int(h, 16))[2:].zfill(B)
        out[i] = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
    return out


def rank(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    # average ties
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def spearman(a, b):
    return pearson(rank(np.asarray(a, float)), rank(np.asarray(b, float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--batches", default="8,16,25,50,100,200")
    ap.add_argument("--draws", type=int, default=64, help="random subsamples per (step,B')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl)]
    rng = np.random.default_rng(a.seed)
    Bs = [int(x) for x in a.batches.split(",")]

    mats = [decode(r["correct_bitmap_hex"], a.B) for r in rows]
    N = mats[0].shape[0]
    print(f"{len(mats)} steps, N={N}, B={a.B}\n")

    res = {}
    for Bp in Bs:
        rho_all, cos_all, dist_all, subq_all = [], [], [], []
        for M in mats:
            f_full = M.mean(axis=1)
            z_full = (f_full - f_full.mean()) / (f_full.std() + 1e-8)
            for _ in range(a.draws if Bp < a.B else 1):
                cols = rng.choice(a.B, size=Bp, replace=False) if Bp < a.B else np.arange(a.B)
                sub = M[:, cols]
                f = sub.mean(axis=1)
                if f.std() < 1e-12:
                    rho_all.append(0.0); cos_all.append(0.0)
                    dist_all.append(1); subq_all.append(1.0)
                    continue
                z = (f - f.mean()) / (f.std() + 1e-8)
                h = Bp // 2
                rho_all.append(spearman(sub[:, :h].mean(axis=1), sub[:, h:2 * h].mean(axis=1)))
                cos_all.append(pearson(z, z_full))
                dist_all.append(len(set(f.tolist())))
                spread = f.max() - f.min()
                subq_all.append(1.0 if spread < 1.0 / Bp else 0.0)
        res[Bp] = dict(rho=float(np.median(rho_all)), rho_mean=float(np.mean(rho_all)),
                       cos=float(np.mean(cos_all)), cos_med=float(np.median(cos_all)),
                       distinct=float(np.median(dist_all)), subquantum=float(np.mean(subq_all)))

    # cost model: generation is the bottleneck and throughput saturates with concurrency.
    # measured: 240 seqs -> 1880 tok/s, 6000 seqs -> 13100 tok/s  =>  fit tput = k * seqs^p
    p = math.log(13100 / 1880) / math.log(6000 / 240)
    k = 13100 / (6000 ** p)
    print(f"throughput model: tok/s = {k:.1f} * seqs^{p:.3f}   (fit to the two measured points)\n")

    def cost(Bp, n=None):
        n = n or N
        seqs = n * Bp
        return seqs / (k * seqs ** p)          # ~ tokens/step / (tok/s), tokens/seq factors out

    c200 = cost(a.B)
    print(f"{'B':>5} {'rho_med':>8} {'cos(z,z200)':>12} {'distinct':>9} {'sub-quantum':>12} "
          f"{'rel_cost':>9} {'rel_s/step':>10} {'fidelity/cost':>14}")
    for Bp in Bs:
        r = res[Bp]
        rc = cost(Bp) / c200
        print(f"{Bp:>5} {r['rho']:8.3f} {r['cos']:12.3f} {r['distinct']:9.0f} "
              f"{r['subquantum']:12.1%} {rc:9.3f} {rc * 194.7:10.1f} {r['cos'] / rc:14.2f}")

    if a.out:
        json.dump({"N": N, "B": a.B, "steps": len(mats), "tput_k": k, "tput_p": p,
                   "results": {str(b): res[b] for b in Bs}}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
