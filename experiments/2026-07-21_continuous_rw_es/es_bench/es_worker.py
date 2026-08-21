"""vLLM WorkerExtension for ES weight surgery.

Design (protocol constraint 6b): keep a RESIDENT PRISTINE base copy of every
parameter in fp32. For each population member, reconstruct the live (fp16/bf16)
weights directly as  theta = base + sigma * eps_i  (a single write, NOT an
incremental fused sigma*(eps_new - eps_old) add/subtract, which drifts in low
precision). The committed ES update accumulates the delta in fp32 on the base
copy, so there is no precision drift across steps.

eps_i is generated per-parameter from a torch.Generator seeded with the member
seed, iterating parameters in a fixed order, in fp32 — so es_set_member and
es_commit_update reconstruct the identical noise.
"""
import torch


class ESWorker:
    # ---- setup ----
    def es_snapshot_base(self):
        self._es_base = {}
        self._es_order = []
        for name, p in self.model_runner.model.named_parameters():
            self._es_base[name] = p.data.detach().to(torch.float32).clone()
            self._es_order.append(name)
        torch.cuda.synchronize()
        return True

    def _named_params(self):
        return self.model_runner.model.named_parameters()

    # ---- per-member reconstruction ----
    def es_set_member(self, seed, sigma):
        sigma = float(sigma)
        for name, p in self._named_params():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
            p.data.copy_((self._es_base[name] + sigma * noise).to(p.dtype))
        torch.cuda.synchronize()
        return True

    def es_restore_base(self):
        for name, p in self._named_params():
            p.data.copy_(self._es_base[name].to(p.dtype))
        torch.cuda.synchronize()
        return True

    # ---- ES update: base += sum_i coeff_i * eps_i ; live weights <- base ----
    def es_commit_update(self, seeds, coeffs):
        seeds = [int(s) for s in seeds]
        coeffs = [float(c) for c in coeffs]
        delta_sq = 0.0
        for name, p in self._named_params():
            base = self._es_base[name]
            delta = torch.zeros_like(base)
            for seed, c in zip(seeds, coeffs):
                if c == 0.0:
                    continue
                gen = torch.Generator(device=p.device)
                gen.manual_seed(seed)
                noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
                delta.add_(noise, alpha=c)
            base.add_(delta)
            p.data.copy_(base.to(p.dtype))
            delta_sq += float(delta.double().pow(2).sum().item())
        torch.cuda.synchronize()
        return delta_sq  # squared L2 norm of the parameter update

    # ---- diagnostics ----
    def es_max_mem(self):
        return {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def es_reset_peak_mem(self):
        torch.cuda.reset_peak_memory_stats()
        return True

    def es_save_base(self, filepath):
        sd = {name: self._es_base[name].to(torch.float16).cpu() for name in self._es_order}
        torch.save(sd, filepath)
        return True
