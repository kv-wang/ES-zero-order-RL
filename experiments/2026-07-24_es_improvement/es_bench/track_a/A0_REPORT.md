# Track A — Phase A0 gate: LoRA-ES build + microbenchmark

**Verdict: PASS. LoRA-ES works and is 2.7–3.6× faster per step than GRPO at matched generation
volume, at lower VRAM.** The A0 KILL criterion (s/step ≥ 2× LoRA-GRPO) is not remotely triggered
— LoRA-ES is 3.6× *faster* than LoRA-GRPO. Proceed to A1 (training sanity).

## Build
LoRA-ES on the in-process vLLM v1 engine: N population members = N LoRA adapters over one frozen
base; each step injects N adapters IN-MEMORY (no disk) via a worker extension
(`_adapter_manager.add_adapter/activate_adapter` + `LoRAModel.from_lora_tensors`, must pass the
manager's `embedding_modules`/`embedding_padding_modules`), reconstructs member adapters = Θ+σ·ε
from seeds on the worker (no tensor passing over RPC), issues ONE batched generate with per-request
`LoRARequest`+seed, and commits the z-score ES update on adapter params only. LoRA cfg: rank 16,
alpha 32, targets = attn{q,k,v,o}+MLP{gate,up,down}; σ_adapter own scale (default 0.01, tuned in A1).

## HARD GATE — adapter isolation: PASSED
Two distinct adapters in one batch each **track their own adapter** (own-agreement 0.2–0.83,
other-agreement ~0.00; first tokens always own; identical-adapter requests agree within the batch).
Finding: vLLM's grouped multi-LoRA punica kernels are *numerically* non-identical to solo runs
(not a leak) — so we assert per-request routing dominance, not bitwise batched==solo. Correct
routing is all ES needs (each member's reward comes from its own adapter in the batch).

## Microbenchmark (Qwen2.5-3B, max_resp/tok 256, 5 warmup + 20 timed, cuda-sync)
LoRA-ES N-curve (all N members in ONE batched generate):

| N | seqs/step | s/step | gen | inject | update |
|---|---|---|---|---|---|
| 4 | 32 | 1.93 | 1.83 | 0.07 | 0.04 |
| 8 | 64 | 2.89 | 2.66 | 0.16 | 0.07 |
| 16 | 128 | 4.10 | 3.75 | 0.27 | 0.09 |
| 30 | 240 | 6.77 | 6.09 | 0.54 | 0.14 |

Head-to-head at **64 seqs/step** (ES N=8 vs GRPO TRAIN_BS=8×G=8):

| method | s/step | gen | update (actor+weights) | ref+logp | **speedup vs LoRA-ES** |
|---|---|---|---|---|---|
| **LoRA-ES (N=8)** | **2.89** | 2.66 | 0.07+0.16 (update+inject) | — | 1.0× |
| full-param GRPO | 7.84 | 1.79 | 1.97+1.31 | 2.14+0.61 | **2.7× slower** |
| LoRA-GRPO | 10.44 | 2.51 | 4.57+1.46 | 0.59+1.29 | **3.6× slower** |

## Findings
1. **LoRA-ES is generation-bound**; adapter-write (inject) and the ES update are cheap and scale
   gently with N. No backward, no optimizer — the whole cost is the (batched) forward generate.
2. **LoRA-GRPO is SLOWER than full-param GRPO in verl** (10.44 vs 7.84 s) — verl's LoRA actor
   update is 2.3× the full-param update (4.57 vs 1.97 s). In verl, LoRA buys VRAM, not wall-clock.
3. **VRAM** (measured peak, nvidia-smi): the number is dominated by each method's configurable
   KV/rollout pool, so compare the *training-state overhead* on top of the shared ~71 GB KV budget
   (util 0.5): LoRA-ES **~71 GB** (KV only, negligible ES overhead) < LoRA-GRPO **84 GB** (+13,
   adapter optimizer/grads/act) < full-param GRPO **103 GB** (+32, full optimizer/grads/act).
   LoRA-ES has the lowest footprint — it carries no training states. (LoRA-ES timing is util-robust:
   2.69 s at util 0.5 vs 2.89 at 0.85.)

## Caveats
- Single config (256 max_tokens, one prompt set); the h-to-plateau × s/step economics is A2's job.
- verl backward/optimizer are not separable (bundled in update_actor) — reported as one number.
- VRAM peaks are KV-elastic; the training-overhead deltas above are the meaningful comparison.
- LoRA-GRPO here uses the same rank/targets as LoRA-ES (rank 16, attn+MLP) for symmetry.

## STOP — A0 gate. Next: A1 (LoRA-ES training sanity, 1.5B, 1 seed, ~300 steps; σ_adapter 3-point
## sweep mirrored by a LoRA-GRPO lr 3-point sweep). Awaiting go-ahead.
