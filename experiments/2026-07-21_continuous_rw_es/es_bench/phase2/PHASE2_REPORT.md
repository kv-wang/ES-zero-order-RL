# Phase 2 — Single-Step Wall-Clock, 3 Arms (Q2)

**Same GPU0, same model (Qwen2.5-3B-Instruct), same shared reward, same
max_tokens=512, all generation via vLLM.** Timing = `torch.cuda.synchronize`
boundaries + `perf_counter`; first 5 warmup steps discarded; ≥25 timed steps;
mean ± std. One-time engine/process startup excluded from s/step.

| arm | seqs/step | s/step (s) | generation | breakdown (other) | tokens/s | peak VRAM |
|---|---|---|---|---|---|---|
| ES N=4, B=8 | 32 | **7.76 ± 0.04** | 7.51 | perturb+swap 0.134, update 0.117 | 1878 | 87995 MiB |
| ES N=8, B=8 | 64 | 15.46 ± … | 15.00 | perturb+swap 0.269, update 0.188 | 1885 | 87993 MiB |
| GRPO (G=8, bs=8) | 64 | **10.18** | 3.03 | logprob-fwd 3.05, actor-update(fwd+bwd+optim) 2.80, overhead ~1.3 | ~2799 | 105063 MiB |

(GRPO actual = 64 rollouts/step, matching the spec's ~64 target. verl's
`update_actor` bundles forward+backward+optimizer; it is not split further.)

## Q2 answer
**Can ES N=4 single-step wall-clock be ≤ GRPO?** **Yes — ES N=4 = 7.76 s < GRPO =
10.18 s per step.** But the two do different work per step: ES N=4 generates 32
sequences, GRPO 64. At **matched generation volume**, ES N=8 (64 seqs) = 15.46 s
is **~1.5× slower** than GRPO (10.18 s).

### Why ES is slower per generated sequence
- ES generation dominates its step (>96%) and runs at ~1880 tok/s; GRPO rollout
  runs at ~2800 tok/s. ES must issue **N separate `generate()` calls of only B=8
  prompts each** (different weights per member ⇒ cannot be batched into one call),
  which under-fills the GPU. GRPO batches all 64 rollouts in a single vLLM call.
- ES's non-generation overhead is genuinely small: perturb+swap (full fp32→fp16
  weight reconstruction per member) is 0.13–0.27 s/step and the committed ES
  update is 0.12–0.19 s/step — i.e. the resident-fp32-base + reconstruct design
  (protocol 6b) adds negligible wall-clock. So ES's cost is essentially all
  under-batched generation.
- Implication: ES's single-step advantage at N=4 comes from doing *less
  generation*, not from being more efficient. Larger B (better batching) or more
  members-per-call would improve ES's tokens/s; with B=8 it is generation-bound.

### VRAM
Raw peaks: ES ≈ 88.0 GB, GRPO ≈ 105.1 GB (ES ~17 GB lower). Note both are inflated
by the vLLM KV-cache reservation (`gpu_memory_utilization=0.5` ⇒ ~71.5 GB
reserved on the 143 GB H200), which is configurable and not intrinsic. ES's
*intrinsic* training memory is just model (fp16) + resident base (fp32) + KV,
with **no gradients / optimizer states / activations**, whereas GRPO additionally
holds FSDP gradients + Adam moments + activations + a separate rollout engine —
which is the ~17 GB difference and the basis of the "ES saves VRAM" thesis.

### Caveats (per spec)
Extrapolated h/seed figures are intentionally **not** reported: steps-to-plateau
was not measured, so per-step numbers must not be turned into end-to-end training
time. This phase measures single-step wall-clock only.

Raw data: `es_N4_summary.json`, `es_N8_summary.json`, `grpo_timing.json`,
`es_N{4,8}.jsonl`, `logs/grpo.log`.
