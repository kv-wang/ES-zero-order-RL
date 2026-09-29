"""ESLoRAWorker + checkpoint I/O and lambda-scaled injection (Phase B/D, overnight run).

Adds three RPCs the base worker lacks:
  es_lora_get_theta()          -> Theta as CPU fp32 tensors, for checkpointing
  es_lora_load_theta(state)    -> restore Theta from such a dict
  es_lora_inject_scaled(lid, lam) -> register an adapter realizing lambda * Delta_W

On the last one: with LoRA, Delta_W = (alpha/r) * B @ A, and Theta_0 has B=0 (verified in D2:
the untrained adapter is an exact no-op, flip-rate 0.000). So the trained displacement from base
is exactly Delta_W, and scaling the B factor by lambda scales the displacement linearly:
    (alpha/r) * (lam*B) @ A = lam * Delta_W
which is the adapter-space analogue of theta_base + lambda*(theta_final - theta_base). This is
what the Phase D shrinkage frontier needs.
"""
from __future__ import annotations
import torch

from es_bench.track_a.src.lora_es_worker import ESLoRAWorker


class LoRAMainWorker(ESLoRAWorker):
    def es_lora_get_theta(self):
        return {k: v.detach().to(torch.float32).cpu() for k, v in self._lora_theta.items()}

    def es_lora_load_theta(self, state):
        for k, v in state.items():
            self._lora_theta[k].copy_(v.to(self._lora_theta[k].device,
                                           dtype=self._lora_theta[k].dtype))
        torch.cuda.synchronize()
        return True

    def es_lora_inject_scaled(self, lid, lam):
        """Register lambda * Delta_W by scaling the B factors only (A untouched)."""
        lam = float(lam)
        tensors = {}
        for key, _ in self._lora_flat:
            t = self._lora_theta[key]
            # keys are '<module>.lora_A.weight' / '.lora_B.weight' (see es_lora_init template)
            tensors[key] = (t * lam if ".lora_B" in key else t).to(self._lora_dtype)
        return self._register(int(lid), tensors)
