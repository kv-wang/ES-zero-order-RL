#!/usr/bin/env python
"""Paired analysis of why trained arms sit below base OOD. No GPU.

Inputs (produced by run_ood_diagnosis.sh):
  es_eval.json + per-question jsonls  : base row + ES theta at lambda {1,.75,.5,.25}, --perq
  grpo_eval.json + per-question jsonls: GRPO hf_merged battery, --perq
  overfit_{base,grpo}.json            : train-pool vs held-out accuracy, per-problem bits

Three questions, three tests:
 1. NOISE?        McNemar on identical problems, arm vs base, per set and pooled OODavg with a
                  proper paired SE. (Unpaired SEs made everything look like noise; paired
                  discordant counts are the real test.)
 2. DRIFT DAMAGE? (ES) OODavg vs lambda dose-response. Random-walk displacement predicts
                  monotone recovery toward base as lambda -> 0 (Phase-1 signature).
 3. MEMORIZATION? (GRPO) difference-in-differences: paired (GRPO-base) delta on its OWN 300
                  train-pool problems minus paired delta on 300 exchangeable held-out problems.
"""
from __future__ import annotations
import argparse, glob, json, math, os
from collections import defaultdict

OOD = ["math500", "svamp", "minerva_math", "olympiadbench", "countdown"]


def perq(prefix, name):
    p = f"{prefix}__{name}_perq.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def mcnemar(base_bits, arm_bits):
    n01 = sum(1 for b, a in zip(base_bits, arm_bits) if not b and a)   # arm wins
    n10 = sum(1 for b, a in zip(base_bits, arm_bits) if b and not a)   # arm loses
    n = len(base_bits)
    delta = (n01 - n10) / n
    var = (n01 + n10) / (n * n) - delta * delta / n     # paired variance of the difference
    se = math.sqrt(max(var, 1e-12))
    k = n01 + n10
    if k == 0:
        p = 1.0
    else:                                                # exact binomial, two-sided
        lo = min(n01, n10)
        cdf = sum(math.comb(k, i) for i in range(lo + 1)) / 2 ** k
        p = min(1.0, 2 * cdf)
    return {"n": n, "arm_wins": n01, "arm_losses": n10, "delta": round(delta, 4),
            "se": round(se, 4), "z": round(delta / se, 2) if se > 0 else 0.0,
            "p_mcnemar": round(p, 4)}


def compare(base_pref, arm_pref):
    per_set = {}
    for name in OOD:
        b, a = perq(base_pref, name), perq(arm_pref, name)
        if b is None or a is None or len(b) != len(a):
            per_set[name] = None
            continue
        per_set[name] = mcnemar([r["correct"] for r in b], [r["correct"] for r in a])
    ok = [v for v in per_set.values() if v]
    d = sum(v["delta"] for v in ok) / len(ok)
    se = math.sqrt(sum(v["se"] ** 2 for v in ok)) / len(ok)
    return {"per_set": per_set,
            "OODavg_delta": round(d, 4), "OODavg_se_paired": round(se, 4),
            "OODavg_z": round(d / se, 2) if se else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = a.dir

    es = json.load(open(f"{D}/es_eval.json"))["entries"]
    base_tag = [t for t in es if "lambda" not in t][0]
    base_pref = f"{D}/es_eval__{base_tag}"

    out = {"paired_vs_base": {}}

    # ---- 1. paired McNemar: each ES lambda row + GRPO, all vs the SAME base perq ----
    for tag in es:
        if "lambda" in tag:
            out["paired_vs_base"][tag] = compare(base_pref, f"{D}/es_eval__{tag}")
    gr = json.load(open(f"{D}/grpo_eval.json"))["entries"]
    grpo_tag = list(gr)[0]
    out["paired_vs_base"]["grpo"] = compare(base_pref, f"{D}/grpo_eval__{grpo_tag}")

    # ---- 2. dose-response: OODavg vs lambda (aggregate accuracies, same protocol) ----
    def oodavg(ev):
        return sum(ev[k]["accuracy"] for k in OOD) / len(OOD)
    dr = {"0.0 (base)": round(oodavg(es[base_tag]), 4)}
    for tag in sorted(es):
        if "lambda" in tag:
            dr[tag.split("lambda")[1]] = round(oodavg(es[tag]), 4)
    out["es_dose_response_OODavg_by_lambda"] = dr

    # ---- 3. GRPO memorization: paired delta on train pool vs on held-out ----
    ob = json.load(open(f"{D}/overfit_base.json"))
    og = json.load(open(f"{D}/overfit_grpo.json"))
    mem = {}
    for side in ["train_pool", "heldout"]:
        mem[side] = mcnemar(ob["sides"][side]["correct"], og["sides"][side]["correct"])
        mem[side]["base_acc"] = ob["sides"][side]["accuracy"]
        mem[side]["grpo_acc"] = og["sides"][side]["accuracy"]
    did = mem["train_pool"]["delta"] - mem["heldout"]["delta"]
    did_se = math.sqrt(mem["train_pool"]["se"] ** 2 + mem["heldout"]["se"] ** 2)
    mem["diff_in_diff"] = {"delta": round(did, 4), "se": round(did_se, 4),
                           "z": round(did / did_se, 2) if did_se else 0.0}
    out["grpo_memorization"] = mem

    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
