#!/usr/bin/env python
"""CPU regression tests for AxisWorker. No GPU, no vLLM, no model download.

The two things worth proving before spending GPU hours:
  1. the baseaxis path is bit-identical to the frozen 2026-07-23 BaseAxisWorker, so the
     baseaxis arm of this try is comparable to base_axis_probe/results/{pilot,confirm};
  2. the momentum path accumulates a correct EMA and reads the axis BEFORE advancing it
     (the buffer aliases the axis, so a naive ordering silently reports post-update
     diagnostics).

Run:  <verl-python> es_bench/axis_probe/tests/test_axis_worker.py
"""
import copy, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRY = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # 2026-08-08_momentum_axis
OLD_TRY = os.path.join(os.path.dirname(TRY), "2026-07-23_lora")


class FakeParam:
    """Minimal stand-in for a torch Parameter as the workers use it."""
    def __init__(self, t):
        self.data = t

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype


def make_params(seed=7):
    g = torch.Generator().manual_seed(seed)
    shapes = {"layer0.w": (6, 5), "layer0.b": (6,), "layer1.w": (4, 6), "layer1.b": (4,)}
    return {k: FakeParam(torch.randn(s, generator=g, dtype=torch.float32)) for k, s in shapes.items()}


class FakeModel:
    def __init__(self, params):
        self._p = params

    def named_parameters(self):
        return list(self._p.items())


class FakeRunner:
    def __init__(self, params):
        self.model = FakeModel(params)


def bind(cls, params):
    """Real code path: the workers reach params through self.model_runner.model."""
    w = cls.__new__(cls)
    w.model_runner = FakeRunner(params)
    w.es_snapshot_base()
    return w


def total_diff(a, b):
    return max(float((a[k].data - b[k].data).abs().max()) for k in a)


def main():
    torch.cuda.synchronize = lambda *args, **kwargs: None   # keep the tests CPU-only

    sys.path.insert(0, TRY)
    from es_bench.axis_probe.src.axis_worker import AxisWorker

    seeds = [11, 22, 33]
    coeffs = [0.03, -0.017, 0.008]
    s_axis = 0.021
    failures = []

    # ---- 1. baseaxis path reproduces the 2026-07-23 worker bit-exactly ----
    # `es_bench` is already bound to this try, so load the frozen worker by file path.
    # It inherits ESWorker from this try, which is md5-identical to the 07-23 copy.
    import importlib.util
    _old = os.path.join(OLD_TRY, "es_bench", "base_axis_probe", "src", "base_axis_worker.py")
    _spec = importlib.util.spec_from_file_location("frozen_base_axis_worker", _old)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseAxisWorker = _mod.BaseAxisWorker

    pa, pb = make_params(), make_params()
    new = bind(AxisWorker, pa); new.es_snapshot_theta0()
    old = bind(BaseAxisWorker, pb); old.es_snapshot_theta0()

    # drift both bases identically so theta0 - theta_t is non-zero
    for w, p in ((new, pa), (old, pb)):
        for k in w._es_order:
            w._es_base[k].add_(torch.full_like(w._es_base[k], 0.05))
            p[k].data.copy_(w._es_base[k])

    n_new = new.es_axis_norm("base")
    n_old = old.es_axis_norm()
    if abs(n_new - n_old) > 1e-9:
        failures.append(f"axis_norm mismatch: {n_new} vs {n_old}")

    new.es_set_axis_member(+1 * 0.4 * 1.0, "base")     # scale = 1 for baseaxis
    old.es_set_anchor_member(0.4, +1)
    if total_diff(pa, pb) > 0:
        failures.append(f"anchor member mismatch: {total_diff(pa, pb)}")

    r_new = new.es_commit_update_axis(seeds, coeffs, s_axis, "base", None)
    r_old = old.es_commit_update_baseaxis(seeds, coeffs, s_axis)
    for key in ("delta_sq", "axis_norm", "signed_disp"):
        if abs(r_new[key] - r_old[key]) > 1e-9:
            failures.append(f"commit {key} mismatch: {r_new[key]} vs {r_old[key]}")
    if total_diff(pa, pb) > 0:
        failures.append(f"post-commit weights mismatch: {total_diff(pa, pb)}")
    if max(float((new._es_base[k] - old._es_base[k]).abs().max()) for k in pa) > 0:
        failures.append("post-commit fp32 base mismatch")
    print(f"[1] baseaxis == 2026-07-23 BaseAxisWorker : {'PASS' if not failures else 'FAIL'}")

    # ---- 2. momentum EMA equals the closed form, and diagnostics are pre-update ----
    beta = 0.9
    pm = make_params()
    mom = bind(AxisWorker, pm); mom.es_mom_init()
    expected = {k: torch.zeros_like(v) for k, v in mom._es_base.items()}
    pre_norms = []

    for step in range(4):
        before = {k: mom._es_base[k].clone() for k in mom._es_order}
        mom_before = {k: mom._mom[k].clone() for k in mom._es_order}
        pre_norm = mom.es_axis_norm("mom")
        pre_norms.append(pre_norm)

        res = mom.es_commit_update_axis(seeds, [c * (step + 1) for c in coeffs],
                                        s_axis if step else 0.0, "mom", beta)
        delta = {k: mom._es_base[k] - before[k] for k in mom._es_order}
        for k in expected:
            expected[k] = beta * expected[k] + delta[k]

        # the reported axis_norm must be the PRE-update momentum, not the advanced one
        if abs(res["axis_norm"] - pre_norm) > 1e-6:
            failures.append(f"step{step}: axis_norm {res['axis_norm']} != pre-update {pre_norm}")
        # signed_disp must project delta on the PRE-update momentum
        if pre_norm > 0:
            dot = sum(float((delta[k].double() * mom_before[k].double()).sum()) for k in delta)
            if abs(res["signed_disp"] - dot / pre_norm) > 1e-6:
                failures.append(f"step{step}: signed_disp aliased to post-update momentum")

    ema_err = max(float((mom._mom[k] - expected[k]).abs().max()) for k in expected)
    if ema_err > 1e-6:
        failures.append(f"EMA mismatch: {ema_err}")
    if pre_norms[0] != 0.0:
        failures.append("momentum should start at zero")
    if not (pre_norms[1] > 0):
        failures.append("momentum should be non-zero after one commit")
    print(f"[2] momentum EMA + pre-update diagnostics    : "
          f"{'PASS' if not any('EMA' in f or 'axis_norm' in f or 'signed_disp' in f or 'momentum' in f for f in failures) else 'FAIL'}")

    # ---- 3. which=None reproduces plain ES (no axis term, no diagnostics) ----
    pv, pw = make_params(), make_params()
    van = bind(AxisWorker, pv)
    ref = bind(AxisWorker, pw)
    van.es_commit_update_axis(seeds, coeffs, 0.0, None, None)
    ref.es_commit_update(seeds, coeffs)
    if total_diff(pv, pw) > 0:
        failures.append(f"vanilla path != ESWorker.es_commit_update: {total_diff(pv, pw)}")
    print(f"[3] which=None == ESWorker.es_commit_update  : "
          f"{'PASS' if not any('vanilla path' in f for f in failures) else 'FAIL'}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all axis-worker tests passed")


if __name__ == "__main__":
    main()
