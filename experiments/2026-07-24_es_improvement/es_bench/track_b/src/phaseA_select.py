#!/usr/bin/env python
"""Phase A auto-selection of sigma* (and alpha = sigma*/2), unattended.

Rule (as specified):
  sigma* = LoRA grid point whose flip-rate is closest to the full-param sigma=1e-3 reference,
           and it must land inside [0.5x, 2x] of that reference.
  alpha  = sigma*/2   (restores alpha/sigma = 0.5; unit-noise update, no 1/sigma -- repo convention)
  If no grid point is inside the band: take the LARGEST sigma, alpha = sigma/2, flag "sigma ceiling".

The probe set is 16 GSM8K + 16 MATH L3-5; each dataset's flip-rate is computed separately and
averaged with equal weight, which is exactly the statistic of the pooled 32-prompt set.
"""
from __future__ import annotations
import argparse, json, glob, os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    refs, loras = {}, {}
    for f in glob.glob(os.path.join(a.dir, "ref_*.json")):
        refs[os.path.basename(f)[4:-5]] = json.load(open(f))
    for f in glob.glob(os.path.join(a.dir, "lora_*.json")):
        loras[os.path.basename(f)[5:-5]] = json.load(open(f))
    if not refs or not loras:
        raise SystemExit(f"missing probe outputs in {a.dir}: refs={list(refs)} lora={list(loras)}")

    def grid(d):
        # probe_sigma_scale.py writes the per-sigma rows under "points"
        return d.get("points") or []

    # pooled reference flip-rate (equal weight across the two 16-prompt halves)
    ref_pts = []
    for ds, d in refs.items():
        g = grid(d)
        if g:
            ref_pts.append(g[0]["flip_rate_mean"])
    ref_flip = sum(ref_pts) / len(ref_pts)

    # pooled LoRA curve
    per_sigma = {}
    for ds, d in loras.items():
        for row in grid(d):
            per_sigma.setdefault(row["sigma"], []).append(row)
    curve = []
    for sig in sorted(per_sigma):
        rows = per_sigma[sig]
        curve.append({
            "sigma": sig,
            "flip_rate": sum(r["flip_rate_mean"] for r in rows) / len(rows),
            "kl_drift": sum(r["kl_drift_mean"] for r in rows) / len(rows),
            "n_datasets": len(rows),
        })

    lo, hi = 0.5 * ref_flip, 2.0 * ref_flip
    in_band = [c for c in curve if lo <= c["flip_rate"] <= hi]
    if in_band:
        best = min(in_band, key=lambda c: abs(c["flip_rate"] - ref_flip))
        sigma_star, ceiling = best["sigma"], False
        reason = (f"flip-rate {best['flip_rate']:.3f} closest to reference {ref_flip:.3f} "
                  f"within band [{lo:.3f},{hi:.3f}]")
    else:
        best = max(curve, key=lambda c: c["sigma"])
        sigma_star, ceiling = best["sigma"], True
        reason = (f"NO grid point in band [{lo:.3f},{hi:.3f}] (flip-rates "
                  f"{[round(c['flip_rate'],3) for c in curve]}); took largest sigma -- SIGMA CEILING")

    out = {"reference_flip_rate_pooled": ref_flip,
           "reference_per_dataset": {ds: grid(d)[0]["flip_rate_mean"] for ds, d in refs.items() if grid(d)},
           "band": [lo, hi], "lora_curve": curve,
           "sigma_star": sigma_star, "alpha": sigma_star / 2.0,
           "sigma_ceiling_flag": ceiling, "selection_reason": reason}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("reference_flip_rate_pooled", "band", "sigma_star", "alpha",
                       "sigma_ceiling_flag", "selection_reason")}, indent=2))
    for c in curve:
        print(f"  sigma={c['sigma']:.4g}  flip={c['flip_rate']:.3f}  kl={c['kl_drift']:.3e}")


if __name__ == "__main__":
    main()
