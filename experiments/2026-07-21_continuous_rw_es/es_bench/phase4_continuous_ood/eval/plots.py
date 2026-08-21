#!/usr/bin/env python
"""Phase 4 figures for REPORT.md. Writes PNGs into results/plots/."""
import json, os, glob
os.environ["MPLCONFIGDIR"] = "/home/hyin66/.cache/p4_mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "results")
P2, P3 = os.path.join(BASE, "phase2"), os.path.join(BASE, "phase3")
OUT = os.path.join(BASE, "plots"); os.makedirs(OUT, exist_ok=True)
LAD = ["svamp", "gsm8k", "minerva_math", "olympiadbench"]
CB, CC = "#4C78A8", "#E45756"  # binary, continuous


def load(d, r, N, s):
    f = f"{d}/{r}_N{N}_s{s}_summary.json"
    return json.load(open(f)) if os.path.exists(f) else None

def oodavg(ev): return float(np.mean([ev[k]["accuracy"] for k in LAD]))

def arm(r, N):
    """(seed rows) preferring phase3; fold phase2 s0 for coverage."""
    rows = []
    for s in (0, 1, 2):
        d = load(P3, r, N, s) or (load(P2, r, N, 0) if s == 0 else None)
        if d is None:
            continue
        ev = d["eval_final"]
        rows.append(dict(ID=ev["math500"]["primary_L3_5_accuracy"], OOD=oodavg(ev),
                         kl=d.get("kl_proxy_drift"), zur=d["zero_update_rate"],
                         ptb=d["pure_tiebreaker_rate"], sstep=d["s_per_step_mean"]))
    return rows

ALLN = [2, 4, 6, 8, 30]
MULTI = [2, 4, 30]

def series(field, Ns):
    out = {"binary": ([], [], []), "continuous": ([], [], [])}
    for r in ("binary", "continuous"):
        for N in Ns:
            rows = arm(r, N)
            vals = [x[field] for x in rows if x[field] is not None]
            if not vals:
                continue
            out[r][0].append(N); out[r][1].append(np.mean(vals))
            out[r][2].append(np.std(vals) if len(vals) > 1 else 0.0)
    return out

# 1) OOD-avg vs N
plt.figure(figsize=(5.2, 3.6))
s = series("OOD", ALLN)
for r, c in (("binary", CB), ("continuous", CC)):
    x, y, e = s[r]
    plt.errorbar(x, y, yerr=e, marker="o", color=c, capsize=3, label=r)
plt.xscale("log"); plt.xticks(ALLN, ALLN); plt.xlabel("population size N")
plt.ylabel("OOD-ladder accuracy"); plt.title("OOD vs N by reward type")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/ood_vs_N.png", dpi=130); plt.close()

# 2) KL proxy vs N (multi-seed)
plt.figure(figsize=(5.2, 3.6))
s = series("kl", MULTI)
for r, c in (("binary", CB), ("continuous", CC)):
    x, y, e = s[r]
    plt.errorbar(x, np.array(y) * 1e3, yerr=np.array(e) * 1e3, marker="o", color=c, capsize=3, label=r)
plt.xscale("log"); plt.xticks(MULTI, MULTI); plt.xlabel("population size N")
plt.ylabel("KL-proxy drift  (×1e3)"); plt.title("Base-drift (KL proxy) vs N")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/kl_vs_N.png", dpi=130); plt.close()

# 3) zero-update (binary) & pure-tiebreaker (continuous) vs N ~ 1/N
plt.figure(figsize=(5.2, 3.6))
zb = series("zur", ALLN)["binary"]
tc = series("ptb", ALLN)["continuous"]
plt.plot(zb[0], zb[1], "o-", color=CB, label="binary zero-update")
plt.plot(tc[0], tc[1], "s-", color=CC, label="continuous pure-tiebreaker")
xs = np.array(ALLN); plt.plot(xs, 1.1 / xs, "k--", alpha=.5, label="~1/N")
plt.xscale("log"); plt.yscale("log"); plt.xticks(ALLN, ALLN); plt.xlabel("population size N")
plt.ylabel("rate"); plt.title("Dead-step rate vs N (tiebreaker rescues them)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{OUT}/rates_vs_N.png", dpi=130); plt.close()

# 4) s/step bars binary vs continuous
plt.figure(figsize=(5.2, 3.6))
sb = series("sstep", ALLN)["binary"]; sc = series("sstep", ALLN)["continuous"]
xi = np.arange(len(ALLN)); w = .38
db = {n: v for n, v in zip(sb[0], sb[1])}; dc = {n: v for n, v in zip(sc[0], sc[1])}
plt.bar(xi - w / 2, [db.get(n, 0) for n in ALLN], w, color=CB, label="binary")
plt.bar(xi + w / 2, [dc.get(n, 0) for n in ALLN], w, color=CC, label="continuous")
plt.xticks(xi, ALLN); plt.xlabel("population size N"); plt.ylabel("s / step")
plt.title("Wall-clock per step (+~5% scoring)"); plt.legend(); plt.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/sstep_vs_N.png", dpi=130); plt.close()

print("wrote:", ", ".join(sorted(os.listdir(OUT))))
