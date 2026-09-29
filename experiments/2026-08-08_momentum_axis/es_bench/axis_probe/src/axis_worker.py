"""vLLM WorkerExtension for the axis-probe family (base-axis and momentum-axis).

Extends the read-only ESWorker with two probeable axes and one unified commit:

  which="base"  theta0 - theta_t   FROZEN original base minus the drifting ES base.
                                   This is the 2026-07-23 base-axis probe. By the identity
                                   theta_t = theta0 + sum_s Delta_s it equals -sum_{s<t} Delta_s,
                                   i.e. the NEGATIVE full-history momentum.
  which="mom"   m_t                EMA momentum buffer, m_t = beta*m_{t-1} + Delta_t.
                                   Variant A1: the same estimator pointed at the RECENT
                                   trajectory instead of the negated full history.

Both axes are probed identically (2 antithetic pairs, central difference, shared z pool), so
an A1-vs-baseaxis comparison isolates the axis and nothing else.

Member perturbation is always  p = base + coef * v  for a scalar coef supplied by the driver;
the driver owns all scale decisions (see es_train_axis.py for the derivation of coef and s_axis).

Memory: "base" holds a frozen fp32 theta0 (~6.2GB @1.5B). "mom" holds one fp32 EMA buffer of the
same size, so the two arms have identical footprints. "vanilla" allocates neither.
"""
import torch

from es_bench.es_worker import ESWorker


class AxisWorker(ESWorker):
    # ---------------- axis storage ----------------

    def es_snapshot_theta0(self):
        """Freeze theta0 = current base. Call right after es_snapshot_base at step 0."""
        self._theta0 = {name: self._es_base[name].clone() for name in self._es_order}
        torch.cuda.synchronize()
        return True

    def es_mom_init(self):
        """Allocate the EMA momentum buffer at zero."""
        self._mom = {name: torch.zeros_like(self._es_base[name]) for name in self._es_order}
        torch.cuda.synchronize()
        return True

    def es_param_count(self):
        return int(sum(self._es_base[name].numel() for name in self._es_order))

    # ---------------- axis access ----------------

    def _axis(self, name, base, which):
        if which == "base":
            return self._theta0[name] - base
        if which == "mom":
            return self._mom[name]
        raise ValueError(f"unknown axis {which!r}")

    def es_axis_norm(self, which="base"):
        """||v|| in fp64 accumulation. Returns 0.0 before the axis exists."""
        if which == "mom" and not hasattr(self, "_mom"):
            return 0.0
        if which == "base" and not hasattr(self, "_theta0"):
            return 0.0
        sq = 0.0
        for name in self._es_order:
            v = self._axis(name, self._es_base[name], which)
            sq += float(v.double().pow(2).sum().item())
        return sq ** 0.5

    # ---------------- probe member ----------------

    def es_set_axis_member(self, coef, which):
        """p = base + coef * v. coef already carries sign, radius and any rescaling."""
        coef = float(coef)
        for name, p in self._named_params():
            base = self._es_base[name]
            p.data.copy_((base + coef * self._axis(name, base, which)).to(p.dtype))
        torch.cuda.synchronize()
        return True

    def es_set_tilt_member(self, seed, sigma, lam, mom_norm):
        """p = base + σ·ξ + λσ·ζ·m̂ where ξ~N(0,I), ζ~N(0,1), m̂ = m/||m||.

        m̂ is never materialized: 1/||m|| folds into the scalar, so the momentum buffer is
        read in place. A separate unit-vector dict would be a third full fp32 copy of the
        model, which does not fit on the card alongside _es_base and _mom.

        Args:
            seed: RNG seed for both ξ and ζ (ζ uses seed+1)
            sigma: perturbation scale
            lam: tilt strength (λ = sqrt(κ·d/(1-κ))); 0 leaves a plain Gaussian
            mom_norm: ||m|| from es_axis_norm("mom"), read before this step's commit
        """
        seed = int(seed)
        sigma = float(sigma)
        lam = float(lam)
        mom_norm = float(mom_norm)

        # Generate scalar ζ ~ N(0,1) first (uses seed+1 to stay independent from ξ).
        # es_commit_update_axis reseeds identically, so the two agree member by member.
        gen_zeta = torch.Generator(device="cpu")
        gen_zeta.manual_seed(seed + 1)
        zeta = torch.randn(1, generator=gen_zeta).item()
        tilt_coef = (lam * sigma * zeta / mom_norm) if (lam != 0.0 and mom_norm > 0.0) else 0.0

        for name, p in self._named_params():
            base = self._es_base[name]

            # Generate ξ ~ N(0,I_d)
            gen = torch.Generator(device=p.device)
            gen.manual_seed(seed)
            xi = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)

            # ε = σ·ξ + (λσζ/||m||)·m
            pert = sigma * xi
            if tilt_coef != 0.0:
                pert.add_(self._mom[name], alpha=tilt_coef)

            p.data.copy_((base + pert).to(p.dtype))
            del xi, pert

        torch.cuda.synchronize()
        return True

    # ---------------- unified commit ----------------

    def es_commit_update_axis(self, gauss_seeds, gauss_coeffs, s_axis, which, mom_beta=None,
                               tilt_params=None):
        """Delta = sum_i gauss_coeff_i * eps_i + s_axis * v,  with v read BEFORE the update.

        For tilt variant: eps_i = sigma*xi_i + lam*sigma*zeta_i*m/||m||, rebuilt from seeds.

        If mom_beta is not None the EMA momentum is advanced with the ACTUAL committed step,
        m <- mom_beta * m + Delta, matching how SGD momentum accumulates its own updates.

        Args:
            tilt_params: dict with {lam, mom_norm} for tilted Gaussian reconstruction

        Returns delta_sq, the axis norm used this step, and the signed projection of Delta on
        the unit axis. Sign semantics differ by axis and are the caller's to interpret:
          which="base": + means moved TOWARD theta0
          which="mom" : + means moved ALONG the recent trajectory (accelerated)
        """
        gauss_seeds = [int(s) for s in gauss_seeds]
        gauss_coeffs = [float(c) for c in gauss_coeffs]
        s_axis = float(s_axis)
        track_mom = mom_beta is not None and hasattr(self, "_mom")
        beta = float(mom_beta) if track_mom else 0.0

        # Tilt reconstruction. Members were drawn as base + sigma*(xi_i + lam*zeta_i*m/||m||);
        # what gets rebuilt here is the sigma-FREE direction, exactly as the vanilla path does,
        # so the estimator's 1/sigma stays implicit and c remains the plain alpha/N*z.
        # The momentum part then collapses into one scalar: zeta_i depends only on the seed and m
        # is shared across members, so sum_i c_i*lam*zeta_i/||m|| is accumulated up front and
        # applied once per tensor instead of N times.
        # ||m|| is the caller's PRE-commit norm; self._mom[name] is advanced only at the end of
        # that name's iteration below, so every read here still sees the matching buffer.
        is_tilt = tilt_params is not None
        lam = float(tilt_params["lam"]) if is_tilt else 0.0
        mom_norm = float(tilt_params["mom_norm"]) if is_tilt else 0.0
        tilt_live = is_tilt and lam != 0.0 and mom_norm > 0.0
        mom_coef = 0.0
        if tilt_live:
            for seed, c in zip(gauss_seeds, gauss_coeffs):
                if c == 0.0:
                    continue
                gen_zeta = torch.Generator(device="cpu")
                gen_zeta.manual_seed(seed + 1)
                zeta = torch.randn(1, generator=gen_zeta).item()
                mom_coef += c * lam * zeta / mom_norm

        delta_sq = axis_sq = dot = 0.0
        for name, p in self._named_params():
            base = self._es_base[name]
            # which="base": the subtraction allocates, so v is already detached from base.
            # which="mom" : v ALIASES self._mom[name], which is advanced at the end of this
            #               iteration -- every read of v must happen before that.
            v = self._axis(name, base, which) if which else None
            delta = torch.zeros_like(base)
            for seed, c in zip(gauss_seeds, gauss_coeffs):
                if c == 0.0:
                    continue
                gen = torch.Generator(device=p.device)
                gen.manual_seed(seed)
                xi = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
                delta.add_(xi, alpha=c)        # eps = xi, coefficient c absorbs sigma
                del xi
            if mom_coef != 0.0:                # every member's tilt term, in one pass
                delta.add_(self._mom[name], alpha=mom_coef)
            if v is not None and s_axis != 0.0:
                delta.add_(v, alpha=s_axis)

            base.add_(delta)
            p.data.copy_(base.to(p.dtype))

            delta_sq += float(delta.double().pow(2).sum().item())
            if v is not None:
                axis_sq += float(v.double().pow(2).sum().item())
                dot += float((delta.double() * v.double()).sum().item())

            if track_mom:                      # last: invalidates v for which="mom"
                self._mom[name].mul_(beta).add_(delta)

        axis_norm = axis_sq ** 0.5
        signed_disp = (dot / axis_norm) if axis_norm > 0 else 0.0
        torch.cuda.synchronize()
        return {"delta_sq": delta_sq, "axis_norm": axis_norm, "signed_disp": signed_disp}
