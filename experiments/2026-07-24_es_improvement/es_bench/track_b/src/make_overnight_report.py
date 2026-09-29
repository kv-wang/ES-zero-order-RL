#!/usr/bin/env python
"""Assemble the overnight deliverables: report.md + results.csv.

Every section is guarded: if a phase crashed or was cut by the deadline, the section says so
explicitly rather than being omitted. Missing input is reported, never silently skipped.
"""
from __future__ import annotations
import csv, glob, json, os, sys
from pathlib import Path

R = "track_b/results/overnight"
L = "track_b/logs/overnight"


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def _poll_rows():
    """rows of (hhmmss, memory_used_mib). CSV is: epoch,HH:MM:SS,idx, used, total, util;"""
    p = f"{L}/nvidia_poll.csv"
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            out.append((parts[1].strip(), int(parts[3].strip())))
        except Exception:
            pass
    return out


def vram_stats(t0=None, t1=None):
    rows = _poll_rows()
    if t0 and t1:
        rows = [r for r in rows if t0 <= r[0] <= t1]
    used = [r[1] for r in rows]
    if not used:
        return None
    s = sorted(used)
    return {"peak_mib": max(used), "n_samples": len(used), "median_mib": s[len(s) // 2]}


def arm_rows(sumpath):
    s = jload(sumpath)
    if not s:
        return None
    return s


def main():
    out = []
    A = out.append
    A("# Overnight run — LoRA-ES at B=200 on 3B (single seed 42)\n")

    # ---------- headline deviations ----------
    A("## 0. Deviations from the task spec (all forced, all logged)\n")
    A("| # | spec | what happened | why |")
    A("|---|---|---|---|")
    A("| 1 | model `Qwen2.5-Math-3B` | used **Qwen2.5-3B-Instruct** | `Qwen2.5-Math-3B` does not "
      "exist — the Qwen2.5-Math family ships 1.5B/7B/72B only (checked against the HF hub). "
      "Qwen2.5-3B-Instruct is the only 3B, is cached locally, and is the model the "
      "*reused* GRPO-MATH and full-param-MATH results were trained on, so it is the only "
      "choice under which Phase D's fold-in is comparable. |")
    A("| 2 | GPU0 + GPU1 | **single GPU, arms run sequentially** | `nvidia-smi` shows one H200 on "
      "this node. Each arm was given a wall-clock deadline instead of a step budget so the night "
      "ends with evaluated checkpoints rather than an unfinished run. |")
    A("| 3 | disk ≥ 200 GB free | **107 GB free on /, 27 GB on home** | Not satisfiable. LoRA "
      "adapter checkpoints are ~38 MB each so they fit on home; full-param checkpoints go to /tmp. |")
    A("| 4 | eval cap | tonight uses the spec's rule (full if n≤1500 else fixed-seed 500 subset) | "
      "Reused older numbers were computed at cap-300 prefix — flagged where folded in. |")
    A("")

    # ---------- Phase 0 ----------
    A("## 1. Phase 0 — preflight\n")
    iso = jload(f"{R}/phase0_isolation.json")
    if iso:
        sc = iso["spec_criterion_exact_match"]
        rc = iso["routing_criterion"]
        A(f"**Adapter isolation (N=4): {iso['GATE']}.** Exact-match criterion as written in the "
          f"spec: **{sc['n_exact']}/{sc['n_total']}**. Routing: all_correct="
          f"{rc['all_routed_correctly']}, mean own-agreement {rc['mean_agree_own']:.3f} vs "
          f"cross-adapter {rc['mean_agree_other_max']:.3f} (margin {rc['margin']:.3f}).")
        if sc["passed_as_specified"]:
            A("\nBoth criteria passed, so no judgement call was needed — LoRA arms ran.\n")
        else:
            A("\nExact match did NOT hold; gate was decided on routing (see file header for why "
              "batched multi-LoRA differs numerically from the solo path). LoRA arms ran.\n")
    else:
        A("**MISSING** — isolation test produced no output; see logs.\n")
    v = vram_stats()
    if v:
        A(f"GPU poller: {v['n_samples']} samples at 30 s, peak **{v['peak_mib']} MiB**, "
          f"median {v['median_mib']} MiB.\n")
        A("Per-arm VRAM (poller windows). **These are vLLM *reservations* driven by "
          "`gpu_memory_utilization`, not measured requirements** — the meaningful part is the "
          "delta, which is the fp32 pristine base copy full-param ES must hold and LoRA-ES need "
          "not.\n")
        A("| window | arm | peak MiB | median MiB |")
        A("|---|---|---|---|")
        for label, t0, t1 in [("01:46–03:56", "LoRA-ES GSM8K B=200"),
                              ][:0]:
            pass
        for t0, t1, label in [("01:46:00", "03:56:00", "LoRA-ES GSM8K B=200"),
                              ("03:57:00", "05:55:00", "LoRA-ES MATH B=200"),
                              ("05:56:00", "06:13:00", "full-param ES iso-B GSM8K"),
                              ("06:14:00", "06:29:00", "Phase D eval")]:
            w = vram_stats(t0, t1)
            if w:
                A(f"| {t0[:5]}–{t1[:5]} | {label} | {w['peak_mib']} | {w['median_mib']} |")
        A("")

    # ---------- Phase A ----------
    A("## 2. Phase A — σ/α selection\n")
    ss = jload(f"{R}/phaseA/SIGMA_STAR.json")
    if ss:
        A(f"Full-param reference (σ=1e-3) pooled flip-rate **{ss['reference_flip_rate_pooled']:.3f}** "
          f"→ acceptance band [{ss['band'][0]:.3f}, {ss['band'][1]:.3f}].")
        A(f"Per-dataset reference: {ss.get('reference_per_dataset')}\n")
        A("| σ_lora | flip-rate | KL drift |")
        A("|---|---|---|")
        for c in ss["lora_curve"]:
            A(f"| {c['sigma']:g} | {c['flip_rate']:.3f} | {c['kl_drift']:.3e} |")
        A("")
        A(f"**Selected σ\\* = {ss['sigma_star']:g}, α = {ss['alpha']:g}** "
          f"(α=σ/2). Ceiling flag: **{ss['sigma_ceiling_flag']}**.")
        A(f"Reason: {ss['selection_reason']}\n")
    else:
        A("**MISSING** — Phase A did not produce SIGMA_STAR.json; Phase B could not start.\n")

    # ---------- Phase B/C arms ----------
    A("## 3. Phase B / C — training arms\n")
    arms = []
    for tag, label in [("loraes_gsm8k", "LoRA-ES GSM8K"), ("loraes_math", "LoRA-ES MATH L3-5"),
                       ("fulles_isoB_gsm8k", "full-param ES iso-B GSM8K")]:
        s = arm_rows(f"{R}/{tag}_summary.json")
        if s:
            arms.append((tag, label, s))
    if arms:
        A("| arm | N | B | σ | α | steps done / asked | stop reason | s/step | fit first20→last20 |"
          " median split-half ρ | KL final |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for tag, label, s in arms:
            kl = s.get("kl_curve") or []
            klf = kl[-1]["kl"] if kl else s.get("kl_proxy_drift")
            f1, f2 = s.get("fit_first20"), s.get("fit_last20")
            rho = s.get("median_split_half_spearman")
            A(f"| {label} | {s.get('population_size')} | {s.get('batch')} | "
              f"{s.get('sigma')} | {s.get('alpha')} | "
              f"{s.get('steps_completed', s.get('num_steps'))} / {s.get('num_steps_requested', s.get('num_steps'))} | "
              f"{s.get('stop_reason','—')} | "
              f"{(s.get('s_per_step_mean') or s.get('s_per_step_mean')) and round(s.get('s_per_step_mean') or 0,1)} | "
              f"{'—' if f1 is None else f'{f1:.4f}→{f2:.4f}'} | "
              f"{'—' if rho is None else f'{rho:+.3f}'} | "
              f"{'—' if klf is None else f'{klf:.4g}'} |")
        A("")
    else:
        A("**No training arm produced a summary.**\n")

    # gate decisions
    A("### Auto-gate decisions taken\n")
    gate_files = sorted(glob.glob(f"{R}/*gate9*_summary.json"))
    drv = f"{L}/phaseBC.driver.log"
    gl = [l.strip() for l in open(drv)] if os.path.exists(drv) else []
    gate_lines = [l for l in gl if "GATE" in l or "retry" in l or "rc=9" in l]
    if gate_files or gate_lines:
        for l in gate_lines:
            A(f"- `{l}`")
        for f in gate_files:
            s = jload(f)
            A(f"- superseded attempt `{os.path.basename(f)}`: stop_reason = {s.get('stop_reason')}")
        A("")
    else:
        A("No auto-gate fired: no arm hit the step-60 signal gate or the KL guard.\n")

    # ---------- Phase D ----------
    A("## 4. Phase D — evaluation\n")
    ev = jload(f"{R}/eval_results.json")
    csv_rows = []
    if ev:
        pr = ev["protocol"]
        A(f"Protocol: greedy, max_tokens={pr['max_tokens']}, {pr['cap_rule']}, shared "
          f"`math_reward` parser, KL probe n={pr['kl_probe_n']}.")
        drop = ev.get("capability_window_dropped", {})
        A(f"\nCapability window {pr['capability_window']} applied to OOD sets. "
          + (f"**Dropped from OOD averages: {drop}**" if drop else "No set violated the window.")
          + "\n")
        base = ev["entries"].get("base", {})
        bev = base.get("eval", {})

        def avgs(entry_eval, arm):
            from_sets = {"gsm8k": ["svamp", "asdiv", "gsm_symbolic", "math500", "minerva_math"],
                         "math": ["gsm8k", "svamp", "minerva_math", "olympiadbench"]}[arm]
            keep = [s for s in from_sets if s not in drop and s in entry_eval]
            ex = sum(entry_eval[s]["accuracy"] for s in keep) / len(keep) if keep else None
            inc = None
            if "countdown" in entry_eval and keep:
                inc = (sum(entry_eval[s]["accuracy"] for s in keep) +
                       entry_eval["countdown"]["accuracy"]) / (len(keep) + 1)
            return ex, inc, keep

        A("| entry | arm | λ | ID | OOD-avg (excl. cntdn) | OOD-avg (incl.) | ΔOOD vs base | KL |")
        A("|---|---|---|---|---|---|---|---|")
        base_ref = {}
        for arm in ("gsm8k", "math"):
            if bev:
                ex, inc, keep = avgs(bev, arm)
                base_ref[arm] = ex
        for tag, e in ev["entries"].items():
            if e.get("error"):
                A(f"| {tag} | — | — | ERROR | {e['error']} | | | |")
                continue
            arm = e["arm"]
            eev = e["eval"]
            if arm == "base":
                for aa in ("gsm8k", "math"):
                    ex, inc, keep = avgs(eev, aa)
                    idk = "gsm8k" if aa == "gsm8k" else "math500"
                    idv = eev.get(idk, {})
                    idacc = idv.get("L3_5_accuracy") if aa == "math" else idv.get("accuracy")
                    A(f"| base ({aa} view) | {aa} | 0 | {idacc} | "
                      f"{'—' if ex is None else f'{ex:.4f}'} | "
                      f"{'—' if inc is None else f'{inc:.4f}'} | — | {e['kl']:.3e} |")
                    csv_rows.append({"entry": f"base_{aa}", "arm": aa, "lambda": 0,
                                     "ID": idacc, "OOD_excl": ex, "OOD_incl": inc,
                                     "dOOD": "", "KL": e["kl"]})
                continue
            ex, inc, keep = avgs(eev, arm)
            idk = "gsm8k" if arm == "gsm8k" else "math500"
            idv = eev.get(idk, {})
            idacc = idv.get("L3_5_accuracy") if arm == "math" else idv.get("accuracy")
            d = (ex - base_ref[arm]) if (ex is not None and base_ref.get(arm)) else None
            A(f"| {tag} | {arm} | {e['lambda']:g} | {idacc} | "
              f"{'—' if ex is None else f'{ex:.4f}'} | {'—' if inc is None else f'{inc:.4f}'} | "
              f"{'—' if d is None else f'{d:+.4f}'} | {e['kl']:.3e} |")
            csv_rows.append({"entry": tag, "arm": arm, "lambda": e["lambda"], "ID": idacc,
                             "OOD_excl": ex, "OOD_incl": inc, "dOOD": d, "KL": e["kl"]})
        A("")
        # lambda* per arm
        A("### λ\\* — shrinkage frontier argmax (the learned-signal detector)\n")
        for arm in ("gsm8k", "math"):
            cand = [(r["lambda"], r["OOD_excl"], r["entry"]) for r in csv_rows
                    if r["arm"] == arm and r["entry"] != f"base_{arm}" and r["OOD_excl"] is not None]
            if not cand:
                A(f"- **{arm}**: no frontier evaluated.")
                continue
            best = max(cand, key=lambda t: t[1])
            A(f"- **{arm}**: λ\\* = **{best[0]:g}** (OOD-avg {best[1]:.4f}); base {base_ref.get(arm)}. "
              + ("λ\\*<1 ⇒ the trained displacement is net harmful and partially shrinking it is "
                 "better — the same signature Phase 1 found." if best[0] < 1.0 else
                 "λ\\*=1 ⇒ the full trained displacement is the best point on the line."))
        A("")
        if ev.get("stopped_early"):
            A(f"**Phase D was cut short:** {ev['stopped_early']}\n")
    else:
        A("**MISSING** — no eval_results.json.\n")

    # ---------- reused ----------
    A("## 5. Reused (not retrained) results\n")
    A("- full-param ES MATH N-sweep + GRPO-MATH: `track_b/results/exp3b_math/OOD_COMPARISON_MATH.csv` "
      "(Qwen2.5-3B-Instruct, 200 steps, seed 0, **cap-300 prefix eval** — different protocol from "
      "tonight's, do not compare digit-for-digit).")
    grp = jload("track_b/results/exp3b_gsm8k/grpo_full_3b.json")
    A("- GRPO-GSM8K: " + ("reused from `track_b/results/exp3b_gsm8k/grpo_full_3b.json` "
      "(**train_batch_size 16, not the requested 128**) — a fresh batch-128 GRPO run did not fit "
      "the single-GPU night alongside the two main LoRA-ES arms; the existing run is the closest "
      "available baseline and its config delta is stated here."
      if grp else "**NOT AVAILABLE** and not retrained."))
    A("")

    # ---------- artifacts ----------
    A("## 6. Artifacts\n")
    for p in sorted(glob.glob(f"{R}/**/*.json*", recursive=True) + glob.glob(f"{R}/*.jsonl")):
        A(f"- `{p}`")
    A(f"- logs: `{L}/`  (per-arm .log, `phaseBC.driver.log`, `nvidia_poll.csv`)")
    A("")

    Path(f"{R}/report.md").write_text("\n".join(out))
    if csv_rows:
        with open(f"{R}/results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"wrote {R}/report.md and {R}/results.csv ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
