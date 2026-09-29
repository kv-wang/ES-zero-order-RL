#!/usr/bin/env python
"""Paper assets from the 3B MATH L3-5 sweep (Phase 0 audit / D3).

Fig A (two panels):
  A1 -- parameter drift vs N against the 1/sqrt(N) diffusion line.
  A2 -- KL per unit squared drift ("coherence cost") vs N, overlaid with the change in
        train fitness. The KL curve lifts off the pure-diffusion line exactly where ES
        starts optimizing its objective (N>=20).

Fig B (two panels) -- the N<=8 destruction regime:
  B1 -- per-step update norm is constant (CV ~1e-4) on every non-zero step, for every N.
  B2 -- member-fitness spread against the B=8 quantization floor (1/B = 0.125).

Outputs: track_b/results/plots/{d3_diffusion_law,destruction_regime}.png + fig_stats.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/hyin66/.cache/p4_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "exp3b_math"
OUT = HERE.parent / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

NS = [4, 8, 20, 30]
KL = {4: 137.1, 8: 54.2, 20: 37.9, 30: 29.1}          # e-3, from OOD_COMPARISON_MATH_REPORT
DL35 = {4: 0.235, 8: 0.332, 20: 0.429, 30: 0.415}      # held-out MATH500 L3-5
BASE_L35 = 0.447
QUANTUM = 0.125                                        # 1/B with B=8

# palette: single accessible sequence, consistent across both figures
C_DATA, C_REF, C_ALT, C_WARN = "#2A6F97", "#8D99AE", "#C1666B", "#D68C45"


def load(n):
    rows = [json.loads(l) for l in open(RES / f"fulles_N{n}.jsonl")]
    u = np.array([r["update_l2"] for r in rows])
    fm = np.array([np.mean(r["primary"]) for r in rows])
    fs = np.array([r["fitness_std"] for r in rows])
    return {"rows": rows, "upd": u, "upd_nz": u[u > 1e-9], "drift": rows[-1]["drift_l2"],
            "fit_delta": float(fm[-20:].mean() - fm[:20].mean()), "fit_std": fs,
            "zero_frac": float(np.mean(u <= 1e-9))}


D = {n: load(n) for n in NS}
stats = {"per_N": {}, "notes": {}}

# ----------------------------------------------------------------- Fig A
drift = np.array([D[n]["drift"] for n in NS], float)
Nv = np.array(NS, float)
klv = np.array([KL[n] for n in NS], float)

# 1/sqrt(N) reference anchored at N=4
ref = drift[0] * np.sqrt(NS[0] / Nv)
resid = drift / ref - 1.0
p_drift = np.polyfit(np.log(Nv), np.log(drift), 1)[0]
p_kl = np.polyfit(np.log(Nv), np.log(klv), 1)[0]
coh = klv / drift**2 * 1e3        # KL per unit squared drift

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

a1 = ax[0]
a1.loglog(Nv, drift, "o-", color=C_DATA, lw=2, ms=8, label="measured drift $\\|\\theta_T-\\theta_0\\|$", zorder=3)
a1.loglog(Nv, ref, "--", color=C_REF, lw=2, label="$1/\\sqrt{N}$ diffusion line (anchored N=4)")
for n, d_, r_ in zip(NS, drift, resid):
    a1.annotate(f"{r_*100:+.1f}%", (n, d_), textcoords="offset points", xytext=(6, 7),
                fontsize=9, color=C_DATA)
a1.set_xlabel("population size $N$"); a1.set_ylabel("parameter drift after 200 steps")
a1.set_xticks(NS); a1.set_xticklabels(NS)
a1.set_xticks([], minor=True)                       # kill 6x10^0-style minor labels
a1.set_ylim(drift.min() * 0.82, drift.max() * 1.30)  # headroom for the residual labels
a1.set_title(f"A1 · drift follows pure diffusion\nfitted exponent {p_drift:.3f} (ideal $-0.5$), "
             f"max deviation {np.abs(resid).max()*100:.1f}%", fontsize=10)
a1.legend(fontsize=8.5, loc="upper right"); a1.grid(alpha=.25, which="both")

a2 = ax[1]
a2.plot(Nv, coh, "o-", color=C_ALT, lw=2, ms=8, label="KL per unit drift$^2$ ($\\times10^{-3}$)", zorder=3)
a2.axhline(coh[:2].mean(), ls="--", color=C_REF, lw=1.8,
           label=f"incoherent-walk level (N$\\leq$8 mean = {coh[:2].mean():.2f})")
a2.set_xlabel("population size $N$"); a2.set_ylabel("KL / drift$^2$  ($\\times10^{-3}$)", color=C_ALT)
a2.tick_params(axis="y", labelcolor=C_ALT); a2.set_xticks(NS)
a2b = a2.twinx()
a2b.bar(Nv, [D[n]["fit_delta"] for n in NS], width=[1.2, 2.0, 4.0, 5.5], alpha=.30,
        color=[C_WARN if D[n]["fit_delta"] < 0 else C_DATA for n in NS], zorder=1)
a2b.axhline(0, color="k", lw=.8)
a2b.set_ylabel("$\\Delta$ train fitness (last 20 − first 20 steps)", fontsize=9)
a2.set_title("A2 · KL lifts off the diffusion line exactly\nwhere train fitness turns positive", fontsize=10)
a2.legend(fontsize=8.5, loc="upper left")
a2.grid(alpha=.25)

fig.suptitle("Full-param ES on Qwen2.5-3B-Instruct, MATH L3-5, 200 steps, seed 0", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "d3_diffusion_law.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Fig B
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

b1 = ax[0]
for n, c in zip(NS, [C_WARN, C_ALT, C_DATA, "#1B4965"]):
    u = D[n]["upd"]
    b1.plot(np.arange(len(u)), u, lw=1.1, color=c,
            label=f"N={n}  (CV={D[n]['upd_nz'].std()/D[n]['upd_nz'].mean():.1e}, "
                  f"{D[n]['zero_frac']*100:.1f}% zero)")
b1.set_xlabel("step"); b1.set_ylabel("update norm $\\|\\Delta\\theta\\|$")
b1.set_title("B1 · the step length carries no signal\nconstant to 4 s.f.; drops to 0 only on tied steps", fontsize=10)
b1.legend(fontsize=8, loc="center right"); b1.grid(alpha=.25)

b2 = ax[1]
parts = [D[n]["fit_std"] for n in NS]
vp = b2.violinplot(parts, positions=range(len(NS)), showmedians=True, widths=.8)
for body in vp["bodies"]:
    body.set_facecolor(C_DATA); body.set_alpha(.45)
for k in ("cbars", "cmins", "cmaxes", "cmedians"):
    vp[k].set_color(C_DATA)
b2.axhline(QUANTUM, ls="--", color=C_WARN, lw=2, label="one fitness quantum $1/B=0.125$")
b2.axhline(QUANTUM / 2, ls=":", color=C_REF, lw=1.8, label="half a quantum")
b2.set_xticks(range(len(NS))); b2.set_xticklabels([f"N={n}" for n in NS])
b2.set_ylabel("member-fitness std within a step")
b2.set_title("B2 · fitness spread sits below the B=8 shot-noise floor\n"
             "ranking decided by a single flipped problem in eight", fontsize=10)
b2.legend(fontsize=8.5); b2.grid(alpha=.25, axis="y")

fig.suptitle("Destruction regime: constant-norm updates + fitness quantization collapse",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "destruction_regime.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- stats
for n in NS:
    d = D[n]
    fs = d["fit_std"]
    stats["per_N"][n] = {
        "drift": d["drift"],
        "drift_vs_invsqrtN_pct": float((d["drift"] / (drift[0] * np.sqrt(NS[0] / n)) - 1) * 100),
        "kl_e3": KL[n],
        "kl_per_drift2_e3": float(KL[n] / d["drift"] ** 2 * 1e3),
        "update_norm_mean_nonzero": float(d["upd_nz"].mean()),
        "update_norm_CV_nonzero": float(d["upd_nz"].std() / d["upd_nz"].mean()),
        "zero_update_frac": d["zero_frac"],
        "fit_std_median": float(np.median(fs)),
        "fit_std_median_in_quanta": float(np.median(fs) / QUANTUM),
        "frac_steps_below_one_quantum": float(np.mean(fs < QUANTUM)),
        "delta_train_fitness": d["fit_delta"],
        "delta_heldout_L3_5": DL35[n] - BASE_L35,
    }
stats["notes"] = {
    "drift_exponent_fitted": float(p_drift), "drift_exponent_ideal": -0.5,
    "kl_exponent_fitted": float(p_kl), "kl_exponent_prereg": -1.0,
    "max_abs_drift_deviation_pct": float(np.abs(resid).max() * 100),
    "kl_times_N": {n: KL[n] * n for n in NS},
    "quantum": QUANTUM, "batch_size": 8, "steps": 200, "seed": 0,
    "model": "Qwen/Qwen2.5-3B-Instruct", "train_set": "MATH L3-5",
}
with open(OUT / "fig_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats["notes"], indent=2))
print("\nwrote:", OUT / "d3_diffusion_law.png", "|", OUT / "destruction_regime.png")
