"""vLLM worker extension for LoRA-ES: in-memory multi-adapter injection + ES commit.

Holds the mean adapter Theta (LoRA A/B for every target module, fp32 on GPU). Each step
reconstructs N member adapters = Theta + sigma*eps_i FROM SEEDS on the worker (no tensor
passing over RPC), registers each as a distinct in-memory LoRA (add_adapter+activate_adapter,
disk path never dereferenced), and returns their lora_int_ids. Commit updates Theta in-place
with the same seeded noise, so inject/commit noise match exactly.

API path (vLLM 0.11, v1, in-process worker): self.model_runner.lora_manager._adapter_manager
-> add_adapter / activate_adapter ; LoRAModel.from_lora_tensors ; PEFTHelper.from_dict.
"""
from __future__ import annotations
import math
import torch


class ESLoRAWorker:
    # ---- init: build Theta (A ~ kaiming, B = 0) ----
    def es_lora_init(self, template, rank, alpha, target_modules, init_seed=0):
        dev = self.model_runner.lora_manager.device
        self._lora_rank = int(rank)
        self._lora_alpha = int(alpha)
        self._lora_targets = list(target_modules)
        self._lora_dtype = self.model_runner.lora_manager.lora_config.lora_dtype
        # flat, fixed-order param list: [(key, shape, is_B), ...] interleaved A,B per module
        self._lora_flat = []
        self._lora_theta = {}
        g = torch.Generator(device=dev); g.manual_seed(int(init_seed))
        for t in template:
            A = torch.empty(tuple(t["shapeA"]), dtype=torch.float32, device=dev)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
            B = torch.zeros(tuple(t["shapeB"]), dtype=torch.float32, device=dev)
            self._lora_theta[t["keyA"]] = A
            self._lora_theta[t["keyB"]] = B
            self._lora_flat.append((t["keyA"], tuple(t["shapeA"])))
            self._lora_flat.append((t["keyB"], tuple(t["shapeB"])))
        self._lora_ntargets = len(template)
        torch.cuda.synchronize()
        return {"n_modules": self._lora_ntargets, "n_tensors": len(self._lora_flat)}

    def _noise(self, seed, j, shape, dev):
        g = torch.Generator(device=dev); g.manual_seed((int(seed) * 1000003 + j) & 0x7FFFFFFFFFFF)
        return torch.randn(tuple(shape), dtype=torch.float32, device=dev, generator=g)

    def _peft_helper(self):
        from vllm.lora.peft_helper import PEFTHelper
        return PEFTHelper.from_dict({"r": self._lora_rank, "lora_alpha": self._lora_alpha,
                                     "target_modules": self._lora_targets,
                                     "peft_type": "LORA", "bias": "none"})

    def _register(self, lid, tensors):
        from vllm.lora.models import LoRAModel
        wm = self.model_runner.lora_manager          # WorkerLoRAManager
        mgr = wm._adapter_manager
        lm = LoRAModel.from_lora_tensors(lora_model_id=int(lid), tensors=tensors,
                                         peft_helper=self._peft_helper(),
                                         device=str(wm.device), dtype=self._lora_dtype,
                                         embedding_modules=wm.embedding_modules,
                                         embedding_padding_modules=wm.embedding_padding_modules)
        mgr.add_adapter(lm)
        mgr.activate_adapter(int(lm.id))
        return int(lm.id)

    # ---- ES: inject N members = Theta + sign_i*sigma*eps(seed_i) ----
    # signs enables antithetic pairs: pass seeds=[s0,s0,s1,s1,...] signs=[+1,-1,+1,-1,...] and the
    # pair (Theta+sigma*eps, Theta-sigma*eps) shares one eps. Commit is unchanged -- the trainer
    # folds the sign into the pair coefficient (z+ - z-) and commits over the unique seeds only.
    def es_lora_inject_members(self, base_id, seeds, sigma, signs=None):
        dev = self.model_runner.lora_manager.device
        sigma = float(sigma)
        if signs is None:
            signs = [1.0] * len(seeds)
        assert len(signs) == len(seeds)
        ids = []
        for i, (seed, sgn) in enumerate(zip(seeds, signs)):
            lid = int(base_id) + i
            s = sigma * float(sgn)
            tensors = {}
            for j, (key, shape) in enumerate(self._lora_flat):
                tensors[key] = (self._lora_theta[key] + s * self._noise(seed, j, shape, dev)).to(self._lora_dtype)
            ids.append(self._register(lid, tensors))
        torch.cuda.synchronize()
        return ids

    # ---- ES commit: Theta += sum_i coeff_i * eps(seed_i) ----
    def es_lora_commit(self, seeds, coeffs):
        dev = self.model_runner.lora_manager.device
        coeffs = [float(c) for c in coeffs]
        upd_sq = 0.0
        for j, (key, shape) in enumerate(self._lora_flat):
            delta = torch.zeros(tuple(shape), dtype=torch.float32, device=dev)
            for seed, c in zip(seeds, coeffs):
                if c == 0.0:
                    continue
                delta.add_(self._noise(seed, j, shape, dev), alpha=c)
            self._lora_theta[key].add_(delta)
            upd_sq += float(delta.double().pow(2).sum().item())
        torch.cuda.synchronize()
        return upd_sq

    def es_lora_theta_norm(self):
        return float(sum(v.double().pow(2).sum().item() for v in self._lora_theta.values())) ** 0.5

    def es_lora_inject_theta(self, lid):
        """Inject the current mean adapter Theta itself (sigma=0) as one adapter, for eval/KL."""
        tensors = {key: self._lora_theta[key].to(self._lora_dtype) for key, _ in self._lora_flat}
        return self._register(int(lid), tensors)

    # ---- generic raw injection (used by the isolation unit test) ----
    def es_lora_inject_raw(self, lid, flat_tensors):
        """flat_tensors: {key: (shape_tuple, flat_float_list)} -> register one adapter."""
        dev = self.model_runner.lora_manager.device
        tensors = {k: torch.tensor(v[1], dtype=self._lora_dtype, device=dev).reshape(v[0]) for k, v in flat_tensors.items()}
        return self._register(lid, tensors)
