#!/usr/bin/env python
"""Build the Phase D evaluation spec from whatever training arms actually produced checkpoints.

Per arm: the FINAL checkpoint gets the full shrinkage frontier (lambda in 0.25/0.5/0.75/1.0 --
lambda=1.0 is the final checkpoint itself, so it is not evaluated twice), and the checkpoint
closest to the best-train-fitness step gets lambda=1.0 only. Arms that produced nothing are
skipped with a printed reason rather than silently dropped.
"""
from __future__ import annotations
import json, os, sys

R = "track_b/results/overnight"
ARMS = [("loraes_gsm8k", "gsm8k"), ("loraes_math", "math")]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]


def main():
    spec = []
    for tag, arm in ARMS:
        sp = f"{R}/{tag}_summary.json"
        if not os.path.exists(sp):
            print(f"SKIP {tag}: no summary ({sp})")
            continue
        s = json.load(open(sp))
        cks = [c for c in s.get("checkpoints", []) if os.path.exists(c["path"])]
        if not cks:
            print(f"SKIP {tag}: no checkpoint files on disk")
            continue
        final = cks[-1]
        spec.append({"tag": f"{tag}_final", "arm": arm, "ckpt": final["path"], "lambdas": LAMBDAS})
        best_step = s.get("best_fit_step")
        if best_step is not None and best_step >= 0:
            num = [c for c in cks if isinstance(c["step"], int)]
            if num:
                nearest = min(num, key=lambda c: abs(c["step"] - best_step))
                if nearest["path"] != final["path"]:
                    spec.append({"tag": f"{tag}_beststep{nearest['step']}", "arm": arm,
                                 "ckpt": nearest["path"], "lambdas": [1.0]})
        print(f"{tag}: steps={s.get('steps_completed')} best_fit_step={best_step} "
              f"ckpts={len(cks)} -> {len([x for x in spec if x['arm']==arm])} entries")
    out = f"{R}/eval_spec.json"
    json.dump(spec, open(out, "w"), indent=2)
    print(f"wrote {out} with {len(spec)} entries")
    if not spec:
        sys.exit(2)


if __name__ == "__main__":
    main()
