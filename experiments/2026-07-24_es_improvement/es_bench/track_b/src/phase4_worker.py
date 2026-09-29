"""Phase 4 vLLM WorkerExtension: ESWorker + a bitwise param/base checksum.

Subclasses the read-only es_bench.es_worker.ESWorker (NO modification to it) and
adds fingerprint RPCs used by the Phase 0c GPU checksum test to prove:
  - es_set_member changes the LIVE weights,
  - the scoring pass does NOT corrupt live weights or the resident fp32 base,
  - es_restore_base returns live weights bit-for-bit to base.

Checksum: two float64 accumulators over params in fixed order — sum and
sum-of-squares in double precision. Cheap, order-stable, sensitive to any change.
"""
from __future__ import annotations
import torch

from es_bench.es_worker import ESWorker


class Phase4Worker(ESWorker):
    def _fingerprint(self, tensors):
        s = ss = 0.0
        for t in tensors:
            td = t.detach().to(torch.float64)
            s += float(td.sum().item())
            ss += float(td.pow(2).sum().item())
        return {"sum": s, "sumsq": ss}

    def es_live_checksum(self):
        """Fingerprint of the live (fp16) model weights, in fixed param order."""
        return self._fingerprint(p.data for _, p in self._named_params())

    def es_base_checksum(self):
        """Fingerprint of the resident fp32 base copy, in fixed param order."""
        return self._fingerprint(self._es_base[name] for name in self._es_order)
