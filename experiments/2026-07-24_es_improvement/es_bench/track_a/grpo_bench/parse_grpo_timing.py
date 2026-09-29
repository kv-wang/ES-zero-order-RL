#!/usr/bin/env python
"""Parse verl console log -> per-step timing_s/* decomposition (mean±std, warmup discarded)."""
import json, re, sys
import numpy as np

WARMUP = 5
KEYS = ["gen", "old_log_prob", "ref", "adv", "update_actor", "update_weights", "step"]


def main():
    log, out = sys.argv[1], sys.argv[2]
    text = open(log, errors="ignore").read()
    series = {}
    for k in KEYS:
        vals = [float(x) for x in re.findall(rf"timing_s/{k}:([-\d.eE]+)", text)]
        series[k] = vals
    n = len(series["step"])
    def stat(k):
        v = series[k][WARMUP:n]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} if v else None
    res = {"n_steps_logged": n, "timed_steps": max(0, n - WARMUP),
           **{f"t_{k}": stat(k) for k in KEYS}}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
