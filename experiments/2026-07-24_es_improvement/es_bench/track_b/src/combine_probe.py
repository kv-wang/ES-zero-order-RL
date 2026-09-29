#!/usr/bin/env python
"""D2 combiner: solve for sigma* -- the LoRA exploration radius whose BEHAVIORAL scale
matches the full-param reference at sigma=1e-3.

Two independent observables are matched (answer flip-rate, per-token KL drift). Each is
inverted separately by log-log interpolation of the LoRA grid, giving sigma*_flip and
sigma*_KL. If they agree the calibration is well-posed; if not, the disagreement itself is
reported rather than averaged away.

Outputs results/d2_probe/D2_SIGMA_STAR.json + results/plots/d2_sigma_calibration.png
"""
from __future__ import annotations
import json, os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/hyin66/.cache/p4_mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "d2_probe"
OUT = HERE.parent / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
REF_SIGMA = 1e-3

fp = json.load(open(RES / "fullparam_ref.json"))
lo = json.load(open(RES / "lora_grid.json"))


def arr(blob, key):
    return (np.array([p["sigma"] for p in blob["points"]], float),
            np.array([p[key] for p in blob["points"]], float),
            np.array([p[key.replace("_mean", "_std")] for p in blob["points"]], float))


ref = next(p for p in fp["points"] if abs(p["sigma"] - REF_SIGMA) < 1e-12)
targets = {"flip_rate_mean": ref["flip_rate_mean"], "kl_drift_mean": ref["kl_drift_mean"]}

res = {"reference": {"arm": "fullparam", "model": fp["model"], "sigma": REF_SIGMA,
                     "flip_rate": ref["flip_rate_mean"], "flip_rate_std": ref["flip_rate_std"],
                     "kl_drift": ref["kl_drift_mean"], "kl_drift_std": ref["kl_drift_std"]},
       "lora_model": lo["model"], "n_prompts": lo["n_prompts"], "n_seeds": lo["n_seeds"],
       "sanity": {"fullparam_self_drift": fp.get("sanity_self_drift"),
                  "lora_theta0_flip": lo.get("sanity_theta0_flip"),
                  "lora_theta0_drift": lo.get("sanity_theta0_drift")},
       "base_extract_rate": {"fullparam": fp.get("base_extract_rate"),
                             "lora": lo.get("base_extract_rate")},
       "lora_grid": [], "sigma_star": {}}

for p in lo["points"]:
    res["lora_grid"].append({k: p[k] for k in
                             ("sigma", "flip_rate_mean", "flip_rate_std",
                              "kl_drift_mean", "kl_drift_std")})


def invert(key, target, logy):
    """Solve grid(sigma) = target for sigma by interpolation in log-sigma."""
    s, y, _ = arr(lo, key)
    o = np.argsort(s); s, y = s[o], y[o]
    ls = np.log10(s)
    yy = np.log10(np.maximum(y, 1e-12)) if logy else y
    tt = np.log10(max(target, 1e-12)) if logy else target
    if not (yy.min() <= tt <= yy.max()):
        return None, ("target %.4g outside grid range [%.4g, %.4g] -- extend the grid"
                      % (target, y.min(), y.max()))
    # monotone-increasing in sigma; np.interp needs increasing x
    return float(10 ** np.interp(tt, yy, ls)), None


for key, logy, label in (("flip_rate_mean", False, "flip"), ("kl_drift_mean", True, "KL")):
    v, err = invert(key, targets[key], logy)
    res["sigma_star"][label] = {"value": v, "target": targets[key], "error": err}

sf = res["sigma_star"]["flip"]["value"]
sk = res["sigma_star"]["KL"]["value"]
if sf and sk:
    res["sigma_star"]["ratio_flip_over_KL"] = sf / sk
    res["sigma_star"]["geometric_mean"] = float(np.sqrt(sf * sk))
    res["sigma_star"]["agree_within_2x"] = bool(0.5 <= sf / sk <= 2.0)

# ------------------------------------------------------------------ plot
C_L, C_F, C_R = "#2A6F97", "#C1666B", "#8D99AE"
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
s, f, fs = arr(lo, "flip_rate_mean")
_, k, ks = arr(lo, "kl_drift_mean")
o = np.argsort(s); s, f, fs, k, ks = s[o], f[o], fs[o], k[o], ks[o]

a0 = ax[0]
a0.errorbar(s, f, yerr=fs, fmt="o-", color=C_L, lw=2, ms=7, capsize=3, label="LoRA grid", zorder=3)
a0.axhline(ref["flip_rate_mean"], ls="--", color=C_F, lw=2,
           label=f"full-param $\\sigma$=1e-3 ref = {ref['flip_rate_mean']:.3f}")
a0.axhspan(ref["flip_rate_mean"] - ref["flip_rate_std"], ref["flip_rate_mean"] + ref["flip_rate_std"],
           color=C_F, alpha=.13)
if lo.get("sanity_theta0_flip") is not None:
    a0.axhline(lo["sanity_theta0_flip"], ls=":", color=C_R, lw=1.8,
               label=f"decoding floor ($\\Theta_0$, no-op) = {lo['sanity_theta0_flip']:.3f}")
if sf:
    a0.axvline(sf, color="k", lw=1.2, ls="-.")
    a0.annotate(f"$\\sigma^*_{{flip}}$={sf:.2e}", (sf, a0.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 14), fontsize=9)
a0.set_xscale("log"); a0.set_xlabel("$\\sigma_{LoRA}$"); a0.set_ylabel("answer flip-rate vs base")
a0.set_title("behavioral match · answer flip-rate", fontsize=10)
a0.legend(fontsize=8); a0.grid(alpha=.25, which="both")

a1 = ax[1]
a1.errorbar(s, np.maximum(k, 1e-9), yerr=ks, fmt="o-", color=C_L, lw=2, ms=7, capsize=3,
            label="LoRA grid", zorder=3)
a1.axhline(ref["kl_drift_mean"], ls="--", color=C_F, lw=2,
           label=f"full-param $\\sigma$=1e-3 ref = {ref['kl_drift_mean']:.2e}")
if sk:
    a1.axvline(sk, color="k", lw=1.2, ls="-.")
    a1.annotate(f"$\\sigma^*_{{KL}}$={sk:.2e}", (sk, a1.get_ylim()[0]),
                textcoords="offset points", xytext=(6, 14), fontsize=9)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("$\\sigma_{LoRA}$"); a1.set_ylabel("per-token KL drift")
a1.set_title("behavioral match · KL drift", fontsize=10)
a1.legend(fontsize=8); a1.grid(alpha=.25, which="both")

fig.suptitle(f"D2 · calibrating $\\sigma_{{LoRA}}$ to the full-param $\\sigma$=1e-3 behavioral scale "
             f"({lo['model'].split('/')[-1]}, {lo['n_prompts']} prompts, {lo['n_seeds']} draws)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "d2_sigma_calibration.png", dpi=170, bbox_inches="tight")
plt.close(fig)

json.dump(res, open(RES / "D2_SIGMA_STAR.json", "w"), indent=2)
print(json.dumps({"reference": res["reference"], "sanity": res["sanity"],
                  "sigma_star": res["sigma_star"]}, indent=2))
print("\nwrote:", RES / "D2_SIGMA_STAR.json", "|", OUT / "d2_sigma_calibration.png")
