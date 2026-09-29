"""Turn verl's console val metrics into the ES arms' evalcurve.jsonl + summary.json shape.

WHY THIS EXISTS
    With save_freq=-1 there is no checkpoint to merge and eval_grpo.py cannot run, but verl's
    own _validate() has already scored the countdown val set against the live rollout weights at
    every test_freq step. Those numbers only exist as console lines, so this reads them back.

WHAT IT READS
    The verl driver log. verl prints one dict per validation, splitting the metric across two
    prefixes (measured on the 6-step smoke, verl 0.6):
        step:0 - val-aux/countdown_es/reward/mean@1:0.3267 - val-core/countdown_es/acc/mean@1:0.3267
    The two are numerically identical because reward_countdown.py is binary. Either IS countdown
    accuracy: it returns countdown_task.answer_reward_function, the same function
    eval_core._eval_countdown uses, and verl's val defaults are greedy (do_sample=False, n=1).

WHAT IT CANNOT PRODUCE
    The six math OOD sets, extract_rate, and kl_proxy_drift. _validate() only scores val.parquet.
    Those keys are omitted rather than written as null, so a reader cannot mistake them for
    measured zeros.

Usage: collect_val_curve.py TRAIN_LOG OUT_PREFIX [--meta k=v ...]
"""
from __future__ import annotations
import json, re, sys

VAL_RE = re.compile(r"val-(?:core|aux)/([^/]+)/([^/]+)/(mean@\d+|[a-z]+@\d+[^:]*):([0-9.eE+-]+)")
STEP_RE = re.compile(r"\bstep:(\d+)\b")

# verl 0.6 splits the countdown metric across two prefixes and the values are identical because
# reward_countdown.py is binary: val-core/countdown_es/acc/mean@1 == val-aux/countdown_es/reward/
# mean@1 (measured: 0.3267/0.3633/0.3700 on the 6-step smoke). Prefer acc -- it is the key verl
# treats as the core metric -- and fall back to reward for older logs that only carry that one.
ACC_SUFFIXES = ("/acc/mean@1", "/reward/mean@1")


def parse(path):
    """Return [{step, countdown, raw:{...}}] in step order, one entry per validation."""
    out, seen = [], set()
    with open(path, errors="replace") as fh:
        for line in fh:
            # Both prefixes, not just val-core: the reward/mean@1 fallback lives under val-aux, so
            # gating on val-core alone made that branch unreachable for a log carrying only it.
            if "val-core/" not in line and "val-aux/" not in line:
                continue
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            metrics = {f"{ds}/{var}/{stat}": float(v) for ds, var, stat, v in VAL_RE.findall(line)}
            if not metrics:
                continue
            # val_before_train logs at the same step index as the first trained step in some verl
            # versions; keep the first occurrence and flag the collision rather than overwrite.
            key = (step, tuple(sorted(metrics)))
            if key in seen:
                continue
            seen.add(key)
            # Record WHICH key supplied the number: the fallback below can pick up an unrelated
            # per-dataset mean after a verl rename, and a bare `countdown: 0.41` would look
            # authoritative. acc_key makes that visible in every evalcurve row.
            acc = acc_key = None
            for suf in ACC_SUFFIXES:
                hit = next((k for k in metrics if k.endswith(suf)), None)
                if hit:
                    acc, acc_key = metrics[hit], hit
                    break
            if acc is None:      # last resort: any per-dataset mean, so a rename degrades to a
                hit = next((k for k in metrics             # wrong-but-VISIBLE number (acc_key
                            if "/mean" in k                # will not be one of ACC_SUFFIXES)
                            and not k.startswith("num_turns/")), None)
                if hit:
                    acc, acc_key = metrics[hit], hit
            out.append({"step": step, "countdown": acc, "acc_key": acc_key, "raw": metrics})
    out.sort(key=lambda r: r["step"])
    return out


def main():
    log, prefix = sys.argv[1], sys.argv[2]
    meta = {}
    for a in sys.argv[3:]:
        if a.startswith("--meta"):
            continue
        k, _, v = a.partition("=")
        try:
            meta[k] = int(v)
        except ValueError:
            try:
                meta[k] = float(v)
            except ValueError:
                meta[k] = v

    rows = parse(log)
    if not rows:
        print(f"!! no val-core/val-aux lines in {log} -- was test_freq>0 set?", file=sys.stderr)
        return 2
    # A verl key rename would leave rows parsed but every countdown None, which used to surface as
    # `max() arg is an empty sequence` from the summary block. Fail here with the actual cause.
    if all(r["countdown"] is None for r in rows):
        keys = sorted({k for r in rows for k in r["raw"]})
        print(f"!! parsed {len(rows)} val lines but no accuracy key matched {ACC_SUFFIXES}.\n"
              f"   keys present: {keys}\n"
              f"   verl likely renamed the metric; update ACC_SUFFIXES.", file=sys.stderr)
        return 3

    with open(f"{prefix}_evalcurve.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    first, last = rows[0], rows[-1]
    summary = {
        "eval_source": "verl_validate_in_memory",
        "eval_note": ("countdown only, greedy, 300 pinned rows == eval_core._eval_countdown's "
                      "[:300] slice, graded by the same binary answer_reward_function. Weights "
                      "were read live from the rollout engine: no checkpoint, no FSDP->HF merge. "
                      "Math OOD sets and kl_proxy_drift are NOT measured by this path."),
        "n_evals": len(rows),
        "acc_keys": sorted({r["acc_key"] for r in rows if r["acc_key"]}),
        "acc_key_expected": all(
            r["acc_key"] and r["acc_key"].endswith(ACC_SUFFIXES) for r in rows),
        "first_eval_step": first["step"],
        "countdown_first": first["countdown"],
        "countdown_final": last["countdown"],
        "final_eval_step": last["step"],
        "countdown_best": max(r["countdown"] for r in rows if r["countdown"] is not None),
        "countdown_best_step": max(
            (r for r in rows if r["countdown"] is not None), key=lambda r: r["countdown"])["step"],
        "curve": [{"step": r["step"], "countdown": r["countdown"]} for r in rows],
        **meta,
    }
    with open(f"{prefix}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  {len(rows)} evals: step {first['step']} -> {last['step']}")
    for r in rows:
        print(f"    step {r['step']:>5}  countdown={r['countdown']}")
    print(f"  wrote {prefix}_evalcurve.jsonl and {prefix}_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
