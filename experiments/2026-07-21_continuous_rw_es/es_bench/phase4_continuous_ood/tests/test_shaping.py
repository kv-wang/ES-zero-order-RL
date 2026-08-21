"""Phase 0c CPU unit tests: shaping correctness + Q1 zero-update replay.

Run: python -m pytest tests/test_shaping.py  (or: python tests/test_shaping.py)
No GPU / vLLM needed. GPU tests (scoring-under-member-weights checksum,
scoring determinism) live in tests/test_scoring_gpu.py.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import continuous_reward as cr  # noqa: E402

PHASE1 = os.path.join(HERE, "..", "..", "phase1")
B = 8


# ---------- lexicographic ordering: primary dominates, secondary breaks ties ----------
def test_delta_below_primary_gap():
    assert cr.tiebreak_delta(B) < 1.0 / B


def test_primary_dominates_worst_case():
    # higher primary by exactly 1/B but with the worst-possible secondary deficit
    f = cr.combined_fitness(primary=[2 / B, 1 / B], secondary=[0.0, 1.0], batch_size=B)
    assert f[0] > f[1]  # the +1/B primary lead survives a full secondary swing


def test_secondary_breaks_exact_primary_tie():
    f = cr.combined_fitness(primary=[0.5, 0.5, 0.5], secondary=[0.1, 0.9, 0.5], batch_size=B)
    # order by f must equal order by secondary when primaries tie
    assert list(np.argsort(f)) == list(np.argsort([0.1, 0.9, 0.5]))


def test_full_lexicographic_order():
    prim = [0.25, 0.5, 0.25, 0.5]
    sec = [0.9, 0.1, 0.3, 0.8]
    f = cr.combined_fitness(prim, sec, B)
    # expected lexicographic (primary, secondary) ascending
    exp = sorted(range(4), key=lambda i: (prim[i], sec[i]))
    assert list(np.argsort(f)) == exp


def test_exact_full_tie_gives_equal_coeffs():
    # identical primary AND secondary -> identical fitness -> identical (zero) coeffs
    f = cr.combined_fitness([0.5, 0.5], [0.3, 0.3], B)
    coeffs, z = cr.zscore_coeffs(f, alpha=5e-4, n=2)
    assert np.allclose(coeffs, coeffs[0])
    assert cr.is_zero_update(f)  # no variance -> zero update


def test_continuous_reduces_to_binary_when_secondary_constant():
    # constant secondary is a pure shift; z-score is shift-invariant => same coeffs as binary
    prim = [0.25, 0.5, 0.75, 0.5]
    f_cont = cr.combined_fitness(prim, [0.42] * 4, B)
    c_cont, _ = cr.zscore_coeffs(f_cont, 5e-4, 4)
    c_bin, _ = cr.zscore_coeffs(prim, 5e-4, 4)
    assert np.allclose(c_cont, c_bin, atol=1e-12)


def test_pure_tiebreaker_gives_nonzero_update():
    # all primaries tie (binary would be zero-update) but secondary varies
    prim = [0.5, 0.5, 0.5, 0.5]
    sec = [0.2, 0.8, 0.5, 0.1]
    assert cr.is_zero_update(prim)                      # binary arm: dead step
    f = cr.combined_fitness(prim, sec, B)
    assert not cr.is_zero_update(f)                     # continuous arm: alive
    assert cr.is_pure_tiebreaker(prim, sec)
    coeffs, _ = cr.zscore_coeffs(f, 5e-4, 4)
    assert np.abs(coeffs).sum() > 0


def test_n_distinct_primary():
    assert cr.n_distinct_primary([0.5, 0.5, 0.5, 0.5]) == 1
    assert cr.n_distinct_primary([0.25, 0.5, 0.5, 0.75]) == 3


# ---------- Q1 zero-update replay: detector must reproduce logged Q1 flags/rates ----------
def _q1_logs():
    return sorted(glob.glob(os.path.join(PHASE1, "es_N*_seed*.jsonl")))


def test_q1_zero_update_replay_flags_match():
    logs = _q1_logs()
    assert logs, "Q1 phase1 jsonl logs not found for replay"
    total_rows = mismatches = 0
    for path in logs:
        for line in open(path):
            row = json.loads(line)
            total_rows += 1
            recomputed = cr.is_zero_update(row["fitness"])
            if recomputed != bool(row["zero_update"]):
                mismatches += 1
    assert mismatches == 0, f"{mismatches}/{total_rows} zero-update flags mismatched"


def test_q1_zero_update_rate_matches_summary():
    for path in _q1_logs():
        summ = path.replace(".jsonl", "_summary.json")
        if not os.path.exists(summ):
            continue
        rows = [json.loads(l) for l in open(path)]
        rate = sum(cr.is_zero_update(r["fitness"]) for r in rows) / len(rows)
        logged = json.load(open(summ))["zero_update_rate"]
        assert abs(rate - logged) < 1e-9, f"{os.path.basename(path)}: {rate} vs {logged}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} shaping/replay tests passed")
