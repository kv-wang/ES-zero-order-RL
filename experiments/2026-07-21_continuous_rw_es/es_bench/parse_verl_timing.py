#!/usr/bin/env python
"""Parse a verl console log into a per-step timing breakdown for the GRPO arm.

Maps verl timing_s/* keys to the spec's GRPO categories:
  generation      = timing_s/gen
  logprob-fwd     = timing_s/old_log_prob + timing_s/ref
  actor update    = timing_s/update_actor   (verl bundles fwd+backward+optimizer;
                                              it is NOT split into backward vs optimizer)
  overhead        = adv + reward + update_weights
Discards the first `warmup` timed steps. Also reports seqs/step and tokens/s.
"""
from __future__ import annotations
import argparse, json, re, sys
import numpy as np

STEP_RE = re.compile(r"step:(\d+)\b")
KV_RE = re.compile(r"(timing_s/\w+|response_length/mean|prompt_length/mean|global_seqlen/mean):([0-9.eE+-]+)")


def parse(logpath, warmup, seqs_per_step):
    per_step = {}
    for line in open(logpath, errors="ignore"):
        if "timing_s/step:" not in line:
            continue
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        d = {k: float(v) for k, v in KV_RE.findall(line)}
        per_step[step] = d
    steps = sorted(per_step)
    timed = [s for s in steps if s > warmup]  # verl steps are 1-indexed; drop first `warmup`
    if not timed:
        return None

    def col(key):
        return np.array([per_step[s].get(key, np.nan) for s in timed], dtype=float)

    gen = col("timing_s/gen")
    lp = col("timing_s/old_log_prob") + np.nan_to_num(col("timing_s/ref"))
    upd = col("timing_s/update_actor")
    adv = np.nan_to_num(col("timing_s/adv"))
    rew = np.nan_to_num(col("timing_s/reward"))
    uw = np.nan_to_num(col("timing_s/update_weights"))
    step_t = col("timing_s/step")
    overhead = adv + rew + uw
    resp_len = col("response_length/mean")

    def st(v):
        v = v[~np.isnan(v)]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if len(v) else None

    out = {
        "n_timed_steps": len(timed), "warmup_dropped": warmup,
        "seqs_per_step": seqs_per_step,
        "s_per_step": st(step_t),
        "t_generation": st(gen),
        "t_logprob_fwd": st(lp),
        "t_actor_update_fwd_bwd_optim": st(upd),
        "t_overhead_adv_reward_syncweights": st(overhead),
        "avg_response_length": st(resp_len),
    }
    if out["s_per_step"] and out["avg_response_length"]:
        toks = out["avg_response_length"]["mean"] * seqs_per_step
        out["tokens_per_step_est"] = toks
        out["tokens_per_s_est"] = toks / out["s_per_step"]["mean"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seqs_per_step", type=int, required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = parse(a.log, a.warmup, a.seqs_per_step)
    if res is None:
        print("NO TIMING DATA PARSED", file=sys.stderr); sys.exit(1)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
