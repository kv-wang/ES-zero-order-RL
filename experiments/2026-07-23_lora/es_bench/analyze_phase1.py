#!/usr/bin/env python
"""Aggregate Phase 1 results:
- ES: per (N, seed) zero-update rate, fitness variance, s/step, t_eff vs N.
- GRPO: zero-advantage-group rate over steps from the reward log.
Writes es_bench/phase1/phase1_summary.json + CSVs + plots (matplotlib, no display).
"""
from __future__ import annotations
import glob, json, os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
from collections import defaultdict
import numpy as np

P1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "es_bench", "phase1")
P1 = os.path.normpath(P1)
GRPO = os.path.join(os.path.dirname(P1), "es_bench", "grpo") if False else \
       os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grpo"))


def load_es():
    runs = {}
    for sf in sorted(glob.glob(os.path.join(P1, "es_N*_seed*_summary.json"))):
        s = json.load(open(sf))
        key = (s["population_size"], s["pop_seed"])
        runs[key] = s
    return runs


def es_table(runs):
    byN = defaultdict(list)
    for (N, seed), s in runs.items():
        byN[N].append(s)
    rows = []
    for N in sorted(byN):
        zs = [s["zero_update_rate"] for s in byN[N]]
        sps = [s["s_per_step"]["mean"] for s in byN[N]]
        fvar = [s["fitness_var_mean"] for s in byN[N]]
        teff = [s["t_eff_per_effective_step"] for s in byN[N]]
        rows.append({
            "N": N, "n_seeds": len(byN[N]),
            "zero_update_rate_mean": float(np.mean(zs)), "zero_update_rate_std": float(np.std(zs)),
            "fitness_var_mean": float(np.mean(fvar)),
            "s_per_step_mean": float(np.mean(sps)),
            "t_eff_mean": float(np.mean(teff)),
            "seeds": [s["pop_seed"] for s in byN[N]],
            "per_seed_zero_rate": zs,
        })
    return rows


def grpo_zero_adv(reward_log, samples_per_step):
    """group by step (gidx//samples_per_step) then uid; zero-adv group = all rewards equal."""
    if not os.path.exists(reward_log):
        return None
    recs = [json.loads(l) for l in open(reward_log) if l.strip()]
    # keep only training-rollout reward calls (exclude any validation calls)
    recs = [r for r in recs if r.get("split") in (None, "train")]
    if not recs:
        return None
    # re-index gidx after filtering so bucketing aligns to training steps
    for i, r in enumerate(recs):
        r["gidx"] = i
    per_step = defaultdict(lambda: defaultdict(list))
    for r in recs:
        step = r["gidx"] // samples_per_step
        per_step[step][r["uid"]].append(r["reward"])
    step_rates = []
    for step in sorted(per_step):
        groups = per_step[step]
        if not groups:
            continue
        zero = sum(1 for uid, rs in groups.items() if len(set(rs)) <= 1)
        step_rates.append({"step": step, "n_groups": len(groups),
                           "zero_adv_groups": zero, "zero_adv_rate": zero / len(groups)})
    overall = float(np.mean([s["zero_adv_rate"] for s in step_rates])) if step_rates else None
    return {"samples_per_step": samples_per_step, "n_steps": len(step_rates),
            "overall_zero_adv_rate": overall, "per_step": step_rates}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_es()
    rows = es_table(runs)
    out = {"es_by_N": rows}

    # GRPO control (train_batch_size * group). Defaults 8*8=64; override via env.
    spb = int(os.environ.get("GRPO_SAMPLES_PER_STEP", "64"))
    rl = os.path.join(GRPO, "reward_phase1_grpo.jsonl")
    ga = grpo_zero_adv(rl, spb)
    out["grpo_zero_adv"] = {k: v for k, v in ga.items() if k != "per_step"} if ga else None

    os.makedirs(os.path.join(P1, "plots"), exist_ok=True)
    json.dump(out, open(os.path.join(P1, "phase1_summary.json"), "w"), indent=2)

    if rows:
        Ns = [r["N"] for r in rows]
        # zero-update rate vs N
        plt.figure(figsize=(6, 4))
        plt.errorbar(Ns, [r["zero_update_rate_mean"] for r in rows],
                     yerr=[r["zero_update_rate_std"] for r in rows], marker="o", capsize=4, label="ES zero-update")
        if out["grpo_zero_adv"] and out["grpo_zero_adv"]["overall_zero_adv_rate"] is not None:
            plt.axhline(out["grpo_zero_adv"]["overall_zero_adv_rate"], ls="--", color="r",
                        label="GRPO zero-adv group rate")
        plt.xlabel("population size N"); plt.ylabel("zero-update / zero-adv rate")
        plt.title("Zero-update rate vs N (Phase 1)"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(P1, "plots", "zero_update_vs_N.png"), dpi=130); plt.close()

        # fitness variance vs N
        plt.figure(figsize=(6, 4))
        plt.plot(Ns, [r["fitness_var_mean"] for r in rows], marker="s")
        plt.xlabel("population size N"); plt.ylabel("mean member-fitness variance")
        plt.title("Member-fitness variance vs N"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(P1, "plots", "fitness_var_vs_N.png"), dpi=130); plt.close()

        # s/step and t_eff vs N
        plt.figure(figsize=(6, 4))
        plt.plot(Ns, [r["s_per_step_mean"] for r in rows], marker="o", label="s/step")
        plt.plot(Ns, [r["t_eff_mean"] for r in rows], marker="^", label="t_eff = s/step / (1-zero_rate)")
        plt.xlabel("population size N"); plt.ylabel("seconds"); plt.legend(); plt.grid(alpha=0.3)
        plt.title("Per-step and effective-step time vs N")
        plt.tight_layout(); plt.savefig(os.path.join(P1, "plots", "sstep_teff_vs_N.png"), dpi=130); plt.close()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
