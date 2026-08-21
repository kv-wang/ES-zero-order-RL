#!/usr/bin/env python
"""Consolidate all phase results into es_bench/FINAL_REPORT.md + report_data.json.
Robust to missing phases — only includes what has been produced."""
from __future__ import annotations
import glob, json, os
B = os.path.dirname(os.path.abspath(__file__))


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def phase0():
    rows = []
    for f in sorted(glob.glob(os.path.join(B, "phase0/results/gsm8k_*_summary.json"))):
        s = load(f); rows.append(("GSM8K", s["model"], s["accuracy"], s["n"]))
    for f in sorted(glob.glob(os.path.join(B, "phase0/results/math_*_summary.json"))):
        s = load(f); rows.append(("MATH L3-5", s["model"], s["accuracy"], s["n"]))
    return rows


def phase1():
    return load(os.path.join(B, "phase1/phase1_summary.json"))


def phase2():
    out = {"es": [], "grpo": None}
    for tag in ["N4", "N8"]:
        s = load(os.path.join(B, f"phase2/es_{tag}_summary.json"))
        if s:
            out["es"].append(s)
    out["grpo"] = load(os.path.join(B, "phase2/grpo_timing.json"))
    peak = None
    pf = os.path.join(B, "phase2/grpo_gpu0_peak_mib.txt")
    if os.path.exists(pf):
        peak = int(open(pf).read().strip() or 0)
    out["grpo_peak_mib"] = peak
    return out


def phase3():
    out = []
    for tag in ["N4", "N8"]:
        s = load(os.path.join(B, f"phase3/dual_{tag}_summary.json"))
        single = load(os.path.join(B, f"phase2/es_{tag}_summary.json"))
        if s:
            speedup = None
            if single:
                speedup = single["s_per_step"]["mean"] / s["s_per_step"]["mean"]
            out.append({"N": s["population_size"], "dual": s,
                        "single_s_per_step": single["s_per_step"]["mean"] if single else None,
                        "speedup_1to2gpu": speedup,
                        "scaling_efficiency": (speedup / 2) if speedup else None})
    return out


def md(data):
    L = ["# ES Small-Population Effectiveness + Wall-Clock Benchmark — Final Report", ""]
    L.append("Model (Phase 1): Qwen2.5-Math-1.5B-Instruct on MATH levels 3-5. "
             "Model (Phase 2/3): Qwen2.5-3B-Instruct. 2x H200. Vanilla ES (no antithetic), "
             "z-score shaping, shared reward object, vLLM everywhere, fp32-base reconstruction.")
    L.append("")

    L.append("## Phase 0 — base accuracy")
    L.append("| task | model | acc | n |")
    L.append("|---|---|---|---|")
    for t, m, a, n in data["phase0"]:
        L.append(f"| {t} | {m.split('/')[-1]} | {a:.3f} | {n} |")
    L.append("")

    p1 = data["phase1"]
    if p1:
        L.append("## Phase 1 — zero-update frequency vs N (Q1)")
        L.append("| N | zero-update rate | fitness var | s/step | t_eff |")
        L.append("|---|---|---|---|---|")
        for r in p1["es_by_N"]:
            L.append(f"| {r['N']} | {r['zero_update_rate_mean']:.3f} ± {r['zero_update_rate_std']:.3f} "
                     f"| {r['fitness_var_mean']:.4f} | {r['s_per_step_mean']:.2f} | {r['t_eff_mean']:.2f} |")
        ga = p1.get("grpo_zero_adv")
        if ga and ga.get("overall_zero_adv_rate") is not None:
            L.append("")
            L.append(f"GRPO control zero-advantage-group rate: **{ga['overall_zero_adv_rate']:.3f}** "
                     f"(over {ga['n_steps']} steps, {ga['samples_per_step']} samples/step).")
        L.append("")

    p2 = data["phase2"]
    if p2 and (p2["es"] or p2["grpo"]):
        L.append("## Phase 2 — single-step wall-clock, same GPU0 + 3B (Q2)")
        L.append("| arm | seqs/step | s/step | generation | perturb-swap / logprob-fwd | update | peak VRAM |")
        L.append("|---|---|---|---|---|---|---|")
        for s in p2["es"]:
            L.append(f"| ES N={s['population_size']} | {s['seqs_per_step']} | "
                     f"{s['s_per_step']['mean']:.2f} | {s['t_generation']['mean']:.2f} | "
                     f"{s['t_perturb_swap']['mean']:.3f} (swap) | {s['t_update']['mean']:.3f} | "
                     f"{s['peak_mem_nvidia_smi_mib']} MiB |")
        g = p2["grpo"]
        if g:
            L.append(f"| GRPO | {g['seqs_per_step']} | {g['s_per_step']['mean']:.2f} | "
                     f"{g['t_generation']['mean']:.2f} | {g['t_logprob_fwd']['mean']:.2f} (logprob-fwd) | "
                     f"{g['t_actor_update_fwd_bwd_optim']['mean']:.2f} (fwd+bwd+optim) | "
                     f"{p2.get('grpo_peak_mib','?')} MiB |")
        L.append("")
        L.append("_GRPO's actor update (verl `update_actor`) bundles forward+backward+optimizer; "
                 "verl does not expose a separate backward vs optimizer split. h/seed extrapolations "
                 "are omitted as steps-to-plateau was not measured (per spec)._")
        L.append("")

    p3 = data["phase3"]
    if p3:
        L.append("## Phase 3 — dual-GPU ES scaling")
        L.append("| N (split) | single-GPU s/step | dual-GPU s/step | speedup | scaling eff |")
        L.append("|---|---|---|---|---|")
        for r in p3:
            sp = f"{r['speedup_1to2gpu']:.2f}x" if r['speedup_1to2gpu'] else "n/a"
            se = f"{r['scaling_efficiency']:.0%}" if r['scaling_efficiency'] else "n/a"
            ss = f"{r['single_s_per_step']:.2f}" if r['single_s_per_step'] else "n/a"
            L.append(f"| {r['N']} ({r['N']//2}+{r['N']//2}) | {ss} | {r['dual']['s_per_step']['mean']:.2f} | {sp} | {se} |")
        L.append("")
    return "\n".join(L)


def main():
    data = {"phase0": phase0(), "phase1": phase1(), "phase2": phase2(), "phase3": phase3()}
    json.dump(data, open(os.path.join(B, "report_data.json"), "w"), indent=2, default=str)
    open(os.path.join(B, "FINAL_REPORT.md"), "w").write(md(data))
    print("wrote FINAL_REPORT.md + report_data.json")
    print(md(data))


if __name__ == "__main__":
    main()
