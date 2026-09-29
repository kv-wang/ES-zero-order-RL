"""vLLM WorkerExtension for ES trajectory replay + weighted checkpoint averaging.

No training and no generation. The ES trajectory is a deterministic function of
  (a) pop_seed  -> the per-step member seed stream (random.Random(pop_seed)), and
  (b) the per-step fitness vectors, which were logged to JSONL during the run,
because es_worker.ESWorker builds eps_i per parameter from a torch.Generator seeded
with the member seed. So every per-step update

    Delta_s = sum_i coeff_i^s * eps(seed_i^s),     coeff = (alpha/N) * zscore(fitness_s)

is reproducible exactly, and no checkpoint files are needed.

The key identity: every quantity Phase 1 needs is a weighted sum of the SAME deltas.
With theta_t = theta_0 + sum_{s<t} Delta_s,

    tail average over checkpoint set T:
        theta_avg = (1/|T|) sum_{t in T} theta_t = theta_0 + sum_s c_s Delta_s,
        c_s = |{t in T : s < t}| / |T|
    pure shrinkage control:
        theta_lambda = theta_0 + lambda * (theta_final - theta_0)

so ONE replay pass that forms Delta_s once per step and scatters it into several
weighted accumulators yields theta_final, any number of tail averages, and (by
rescaling accumulator 0) the whole shrinkage family -- at the cost of the original
run's update phase only (~2 min for 200 steps at N=30, vs ~3.2 h to retrain).

Accumulator 0 is by convention the unweighted sum (c_s = 1), i.e. theta_final - theta_0.
"""
from __future__ import annotations
import math
import torch

from es_bench.track_b.src.phase4_worker import Phase4Worker


class TailAvgWorker(Phase4Worker):
    # ---- accumulators ----
    def es_accum_init(self, n_acc: int):
        """Allocate n_acc fp32 accumulators shaped like the resident base copy."""
        self._acc = [{name: torch.zeros_like(self._es_base[name]) for name in self._es_order}
                     for _ in range(int(n_acc))]
        torch.cuda.synchronize()
        return True

    def es_accum_free(self):
        self._acc = None
        torch.cuda.empty_cache()
        return True

    # ---- replay ----
    def es_replay_chunk(self, chunk):
        """chunk: list of (seeds, coeffs, acc_weights) -- one entry per step.

        Reproduces Delta_s exactly as es_commit_update would have (same generator,
        same per-parameter iteration, fp32), but adds it into the accumulators
        instead of the resident base, so the base copy stays at theta_0.

        Returns the per-step squared L2 norm of Delta_s, for validation against the
        `update_l2` logged during the original run.
        """
        out = []
        for seeds, coeffs, wts in chunk:
            seeds = [int(s) for s in seeds]
            coeffs = [float(c) for c in coeffs]
            wts = [float(w) for w in wts]
            delta_sq = 0.0
            for name, p in self._named_params():
                base = self._es_base[name]
                delta = torch.zeros_like(base)
                for seed, c in zip(seeds, coeffs):
                    if c == 0.0:
                        continue
                    gen = torch.Generator(device=p.device)
                    gen.manual_seed(seed)
                    noise = torch.randn(p.shape, dtype=torch.float32, device=p.device,
                                        generator=gen)
                    delta.add_(noise, alpha=c)
                delta_sq += float(delta.double().pow(2).sum().item())
                for j, w in enumerate(wts):
                    if w != 0.0:
                        self._acc[j][name].add_(delta, alpha=w)
            out.append(delta_sq)
        torch.cuda.synchronize()
        return out

    # ---- materialize a variant ----
    def es_set_theta(self, weights):
        """live weights <- theta_0 + sum_j weights[j] * acc[j].

        The resident fp32 base is never modified, so variants can be materialized in
        any order from the same replay. Returns ||theta - theta_0||_2.
        """
        weights = [float(w) for w in weights]
        dsq = 0.0
        for name, p in self._named_params():
            base = self._es_base[name]
            delta = torch.zeros_like(base)
            for j, w in enumerate(weights):
                if w != 0.0:
                    delta.add_(self._acc[j][name], alpha=w)
            dsq += float(delta.double().pow(2).sum().item())
            p.data.copy_((base + delta).to(p.dtype))
        torch.cuda.synchronize()
        return math.sqrt(max(dsq, 0.0))
