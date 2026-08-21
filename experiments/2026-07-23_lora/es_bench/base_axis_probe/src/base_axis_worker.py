"""vLLM WorkerExtension for the base-axis probe.

Extends the read-only ESWorker. Adds a FROZEN original-base snapshot theta0 (distinct
from _es_base, which drifts as ES commits updates), anchor-member reconstruction along
the base axis (theta0 - theta_t), and a mixed commit that fuses Gaussian noise members
with the antithetic anchor contribution, returning the axis-projection diagnostics.
"""
from __future__ import annotations
import torch

from es_bench.es_worker import ESWorker


class BaseAxisWorker(ESWorker):
    # ---- frozen original base ----
    def es_snapshot_theta0(self):
        """Freeze theta0 = current base (call right after es_snapshot_base at step 0)."""
        self._theta0 = {name: self._es_base[name].clone() for name in self._es_order}
        torch.cuda.synchronize()
        return True

    def es_param_count(self):
        return int(sum(self._es_base[name].numel() for name in self._es_order))

    def es_axis_norm(self):
        """||theta0 - theta_t|| in fp32 (theta_t == current _es_base)."""
        sq = 0.0
        for name in self._es_order:
            sq += float((self._theta0[name] - self._es_base[name]).double().pow(2).sum().item())
        return sq ** 0.5

    # ---- anchor member: theta_t + sign*a*(theta0 - theta_t) ----
    def es_set_anchor_member(self, a, sign):
        a = float(a) * float(sign)
        for name, p in self._named_params():
            base = self._es_base[name]
            axis = self._theta0[name] - base
            p.data.copy_((base + a * axis).to(p.dtype))
        torch.cuda.synchronize()
        return True

    # ---- mixed commit: Gaussian noise members + anchor scalar along the axis ----
    def es_commit_update_baseaxis(self, gauss_seeds, gauss_coeffs, s_anchor):
        """Delta = sum_i gauss_coeff_i * eps_i  +  s_anchor * (theta0 - theta_t).
        Returns (delta_sq, axis_norm, signed_axis_displacement)."""
        gauss_seeds = [int(s) for s in gauss_seeds]
        gauss_coeffs = [float(c) for c in gauss_coeffs]
        s_anchor = float(s_anchor)
        delta_sq = axis_sq = dot = 0.0
        for name, p in self._named_params():
            base = self._es_base[name]
            axis = self._theta0[name] - base            # computed BEFORE update
            delta = torch.zeros_like(base)
            for seed, c in zip(gauss_seeds, gauss_coeffs):
                if c == 0.0:
                    continue
                gen = torch.Generator(device=p.device); gen.manual_seed(seed)
                noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
                delta.add_(noise, alpha=c)
            if s_anchor != 0.0:
                delta.add_(axis, alpha=s_anchor)
            base.add_(delta)
            p.data.copy_(base.to(p.dtype))
            delta_sq += float(delta.double().pow(2).sum().item())
            axis_sq += float(axis.double().pow(2).sum().item())
            dot += float((delta.double() * axis.double()).sum().item())
        axis_norm = axis_sq ** 0.5
        signed_disp = (dot / axis_norm) if axis_norm > 0 else 0.0
        return {"delta_sq": delta_sq, "axis_norm": axis_norm, "signed_disp": signed_disp}
