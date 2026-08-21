# Phase 3 — Dual-GPU ES Scaling + GRPO Multi-GPU Audit

## Dual-GPU ES (Qwen2.5-3B-Instruct, MATH L3–5, max_tokens=512, 30 steps, warmup 5)

One process per GPU (`CUDA_VISIBLE_DEVICES` isolation), population split evenly.
The coordinator (multiprocessing + queues) only distributes member **seeds** and
collects **scalar fitness**; each worker reconstructs θ=base+σε for its members,
evaluates, and applies the full `Σ_i coeff_i·ε_i` update to its own fp32 base.
Because all seeds+coeffs are shared and the noise is deterministic per seed, both
base copies stay bit-identical with **no inter-GPU weight transfer** (contrast the
repo's NCCL weight broadcast).

| N (split) | single-GPU s/step | dual-GPU s/step | speedup | scaling efficiency | peak VRAM/GPU |
|---|---|---|---|---|---|
| 4 (2+2) | 7.76 | 3.95 | **1.97×** | **98%** | ~89 GB |
| 8 (4+4) | 15.46 | 7.84 | **1.97×** | **99%** | ~89 GB |

(single-GPU baselines reused from Phase 2, same model/settings.)

**Result:** ES achieves near-linear (≈98–99%) 1→2 GPU scaling. The dual-GPU step
time = the slower worker's population-eval barrier (3.83 / 7.6 s) + a tiny serial
update (~0.12 s). The only coordination is exchanging N seeds and N fitness
scalars per step — negligible bandwidth — so ES data-parallelism over the
population is essentially communication-free and scales out cleanly to more GPUs
(each additional GPU takes another N/G-th of the population).

## GRPO multi-GPU support audit (verl) — no implementation, per spec

**Already supported natively; a config change, not a code change.** verl's PPO/GRPO
trainer runs the actor (and ref) under **PyTorch FSDP** orchestrated by **Ray**,
with the vLLM rollout engine co-located (hybrid engine). Relevant knobs:
- `trainer.n_gpus_per_node` / `trainer.nnodes` — set GPUs; FSDP shards the actor
  across them (our runs used `n_gpus_per_node=1`; the FSDP logs showed
  `ShardingStrategy=NO_SHARD` precisely because only 1 GPU was visible — with 2 it
  sharding-shards params/grads/optimizer).
- `actor_rollout_ref.rollout.tensor_model_parallel_size` — TP for the rollout
  engine (or data-parallel rollout replicas).
- FSDP variants (`fsdp`/`fsdp2`) and Megatron backend are both available.

**Estimated cost of enabling dual-GPU GRPO:** ~zero engineering — flip
`n_gpus_per_node=2` (and optionally `rollout.tensor_model_parallel_size`/
`ppo_micro_batch_size_per_gpu`). **However**, unlike ES it is *not*
communication-free: each step FSDP performs all-gather (forward) + reduce-scatter
(backward) of parameters/gradients and then reshards updated weights into the
vLLM engine (`timing_s/update_weights`). So GRPO's realized 1→2 GPU efficiency
will be **below** ES's ~98% due to gradient/weight-sync traffic — the exact figure
would need a measured run (out of scope this phase).

**Contrast (the VRAM/scaling thesis):** ES scales the *population* (embarrassingly
parallel, scalar-only communication, no gradients/optimizer to shard), whereas
GRPO scales a *sharded gradient computation* (collective comms every step). This
is the structural reason ES is both lower-memory (Phase 2) and cleaner-scaling
(Phase 3), at the cost of per-step generation efficiency (Phase 2) and a higher
ineffective-step rate at small N (Phase 1).

Raw data: `dual_N{4,8}_summary.json`, `dual_N{4,8}.jsonl`.
