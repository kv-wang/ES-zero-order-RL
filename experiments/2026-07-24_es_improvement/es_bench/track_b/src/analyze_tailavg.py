#!/usr/bin/env python
"""Phase 1 analysis: did tail averaging extract a direction, or just step back?

For each arm the shrinkage family theta_lambda = theta_0 + lambda*(theta_final-theta_0)
traces a one-dimensional frontier of (KL, accuracy). Pure shrinkage extracts NO signal by
construction -- it only walks back along the line already travelled. So the question is
purely positional: does the tail-averaged point sit ABOVE that frontier?

Two comparisons, strongest first:
  1. matched-norm  -- shrink_matched_* was built at exactly ||A_j||, so it differs from the
     averaging arm only in DIRECTION. No interpolation, no modelling. This is the test.
  2. KL-interpolated -- the shrinkage frontier interpolated (linear in log KL) to the
     averaging arm's KL, as a cross-check that the matched-norm point isn't an outlier.

Usage: python analyze_tailavg.py results/phase1_tailavg/*_summary.json
"""
from __future__ import annotations
import json, math, sys

OOD_SETS = ["svamp", "minerva_math", "olympiadbench", "math500", "countdown"]
BASE = {"math500_L3_5": 0.447, "OOD_avg": 0.375}   # exp3b_math base_3b.json, cap 300


def metrics(ev):
    if not ev:
        return {}
    m = {"math500_L3_5": ev["math500"]["primary_L3_5_accuracy"]}
    vals = [ev[k]["accuracy"] for k in OOD_SETS if k in ev]
    m["OOD_avg"] = sum(vals) / len(vals) if len(vals) == len(OOD_SETS) else float("nan")
    for k in ("gsm8k", "svamp", "minerva_math", "olympiadbench", "amc23", "countdown", "math500"):
        if k in ev:
            m[k] = ev[k]["accuracy"]
    return m


def interp_logkl(points, kl, key):
    """Linear interpolation of `key` vs log(KL) across the shrinkage frontier."""
    pts = sorted((p for p in points if p["kl"] > 0 and key in p["m"]), key=lambda p: p["kl"])
    if len(pts) < 2:
        return None
    x = math.log(max(kl, 1e-12))
    if x <= math.log(pts[0]["kl"]):
        lo, hi = pts[0], pts[1]
    elif x >= math.log(pts[-1]["kl"]):
        lo, hi = pts[-2], pts[-1]
    else:
        lo, hi = next((pts[i], pts[i + 1]) for i in range(len(pts) - 1)
                      if math.log(pts[i]["kl"]) <= x <= math.log(pts[i + 1]["kl"]))
    x0, x1 = math.log(lo["kl"]), math.log(hi["kl"])
    if x1 == x0:
        return lo["m"][key]
    t = (x - x0) / (x1 - x0)
    return lo["m"][key] + t * (hi["m"][key] - lo["m"][key])


def main(paths):
    for path in paths:
        s = json.load(open(path))
        if s.get("error"):
            print(f"\n## {path}: FAILED ({s['error']})\n")
            continue
        V = s["variants"]
        N = s["N"]
        print(f"\n## N={N}  ({s['jsonl'].split('/')[-1]})")
        print(f"replay validated: max rel.err vs logged update_l2 = "
              f"{s['replay']['max_rel_err_vs_logged_update_l2']:.1e}   "
              f"lambda_eff = " + ", ".join(f"{k}:{v:.3f}" for k, v in s["lambda_eff"].items()))

        rows = []
        for tag, v in V.items():
            m = metrics(v.get("eval"))
            rows.append({"tag": tag, "kind": v["kind"], "drift": v["drift_l2"],
                         "kl": v["kl"], "m": m})
        shrink = [r for r in rows if r["tag"].startswith("shrink")]
        avgs = [r for r in rows if r["tag"].startswith("tailavg")]
        final = next(r for r in rows if r["tag"] == "theta_final")

        print(f"\n| variant | drift | KL(e-3) | MATH500 L3-5 | OOD-avg |")
        print(f"|---|---|---|---|---|")
        print(f"| base (theta_0) | 0.0 | 0.0 | {BASE['math500_L3_5']:.3f} | {BASE['OOD_avg']:.3f} |")
        for r in sorted(rows, key=lambda r: r["drift"]):
            id_ = r["m"].get("math500_L3_5")
            oo = r["m"].get("OOD_avg")
            print(f"| {r['tag']} | {r['drift']:.1f} | {r['kl']*1e3:.2f} | "
                  f"{'—' if id_ is None else f'{id_:.3f}'} | "
                  f"{'—' if oo is None else f'{oo:.3f}'} |")

        frontier = shrink + [final]
        print(f"\n**Averaging vs shrinkage**")
        print(f"\n| averaging arm | metric | avg | matched-norm shrink | gap | KL-interp shrink | gap |")
        print(f"|---|---|---|---|---|---|---|")
        for av in avgs:
            acc_name = av["tag"].replace("tailavg_", "")
            match = next((r for r in shrink if r["tag"] == f"shrink_matched_{acc_name}"), None)
            for key, label in (("math500_L3_5", "MATH500 L3-5"), ("OOD_avg", "OOD-avg")):
                a = av["m"].get(key)
                if a is None:
                    continue
                mv = match["m"].get(key) if match else None
                iv = interp_logkl(frontier, av["kl"], key)
                f_ = lambda x: "—" if x is None else f"{x:+.3f}"
                g_ = lambda x: "—" if x is None else f"{a-x:+.3f}"
                print(f"| {acc_name} | {label} | {a:.3f} | "
                      f"{'—' if mv is None else f'{mv:.3f}'} | {g_(mv)} | "
                      f"{'—' if iv is None else f'{iv:.3f}'} | {g_(iv)} |")
    print("\n_Noise: single seed, cap-300 eval. MATH500 L3-5 (n=367 capped to 300) SE ~ ±0.028; "
          "OOD-avg (5 sets) SE ~ ±0.013. A gap must clear these to mean anything._")


if __name__ == "__main__":
    main(sys.argv[1:])
