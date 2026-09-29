# Strategy: can ES match GRPO OOD at lower wall-clock / VRAM?

## 1. What the GSM8K data actually shows (honest)
Per-dataset (train=gsm8k, 1.5B-Instruct, vs base; ES = full-param σ=2e-3):

| dataset | n | extract | base | ES | GRPO | ES−base | GRPO−base | ES−GRPO |
|---|---|---|---|---|---|---|---|---|
| gsm8k (ID) | 300 | 0.99 | 0.750 | 0.710 | 0.780 | −0.040 | **+0.030** | −0.070 |
| svamp | 300 | 0.98 | 0.817 | 0.847 | 0.853 | +0.030 | +0.037 | −0.007 |
| minerva | 272 | 0.58 | 0.081 | 0.107 | 0.110 | +0.026 | +0.029 | −0.004 |
| olympiad | 300 | 0.68 | 0.113 | 0.133 | 0.120 | +0.020 | +0.007 | **+0.013** |
| math500 | 300 | 0.76 | 0.443 | 0.403 | 0.427 | −0.040 | −0.017 | −0.023 |

**The real finding is a DISSOCIATION, not "ES > GRPO":**
- **GRPO learns the task** (gsm8k +0.030) and transfers; its gains are fit-driven.
- **ES does NOT learn the task** (gsm8k −0.040) yet still lifts the hard OOD sets (minerva +0.026, olympiad +0.020) about as much as GRPO — its gains are **exploration/drift-driven**, not fit-driven.
- ES nominally beats GRPO only on **olympiad (+0.013)**, but that set has extract-rate 0.68, single seed → within noise. Do NOT over-claim "ES generalizes better" yet.

## 2. Why ES generalizes differently (mechanism)
- ES = zeroth-order, weight-space Gaussian exploration + z-scored averaging → a **flat-minima / regularization** bias with **high drift** (KL 8–18e-3 vs GRPO's KL-penalized ~low). It moves the model broadly *without precisely fitting the reward* → can elicit latent capability on hard OOD while disrupting the sharp ID skill.
- GRPO = first-order policy gradient + KL penalty → **fits the training reward, stays near base**, transfers via genuine task learning.
- So GRPO "learns to solve gsm8k"; ES "perturbs toward a broadly-different model." The OOD lift without ID gain is the signature of drift/exploration, and could partly be format/verbosity change rather than reasoning — needs controls to separate.

## 3. The gating insight for the goal
Goal = **match GRPO OOD at lower cost.** ES's OOD lift currently comes bundled with **ID degradation + high drift**. To be competitive, ES must lift OOD **without** wrecking ID — i.e. control the drift. Three levers:
1. **Right task**: train where ES actually learns (MATH L3-5, base ~0.5, real headroom — ES learned there in Phase 4; on gsm8k base 0.75 the binary signal is too weak/adversarial).
2. **Drift control**: lower σ / KL-to-base penalty / the **base-axis anchor** (ties back to that probe) → keep exploration benefit, stop ID collapse.
3. **Cost tricks (N=4, LoRA)** only pay off AFTER ES is competitive — they reduce compute, not fix learning.

## 4. Experiment plan (phased)
**Exp 1 — ENABLING: 3-way OOD comparison on MATH L3-5 (make-or-break).**
Train MATH L3-5; eval ID=MATH500(L3-5), OOD={gsm8k, svamp, minerva, olympiad}. Methods: full-param ES, LoRA-ES, GRPO (+LoRA-GRPO). N-sweep for ES {4,8,16} to find the smallest N matching GRPO OOD. 1.5B first, 3 seeds on finalists. **If ES matches GRPO OOD here, the whole goal is viable; if not, rethink.**

**Exp 2 — COST PAYOFF: N=4 LoRA-ES vs GRPO at matched OOD.**
Take the min-N ES config from Exp 1; measure wall-clock (h-to-plateau × s/step) + VRAM (matched KV budget) at matched OOD. N=4 LoRA-ES = 4 adapters, one batched generate, forward-only, zero optimizer state → the cheapest ES.

**Exp 3 — SCALING: 3B then 7B.**
The ES advantage GROWS with size: GRPO needs model+gradients+Adam(2× fp32 master/moments)+rollout; ES needs only forward. At 7B, GRPO's optimizer/grad memory dominates while ES stays flat. Show VRAM(GRPO) − VRAM(ES) widening, and wall-clock advantage from forward-only + generation batching.

**Exp 4 — PARALLEL-GPU ADVANTAGE (cheap, do early).**
ES is **comms-free** data-parallel over the population (Phase 3 measured **1.97× on 2 GPUs, 98–99% eff** via es_train_dual.py). GRPO (FSDP) needs gradient all-reduce every step → comms-bound, sub-linear. Demo ES 1→2 GPU speedup vs GRPO 1→2 GPU speedup on 3B/7B. This is a clean ES win independent of the accuracy question.

## 5. Metrics (unified)
ID + full OOD battery + retention ratios · KL-to-base · s/step + steps-to-plateau + total GPU-h · VRAM at matched KV budget · parallel scaling efficiency (1→2 GPU). Unified reward; 3-seed for conclusions.

## 6. Recommended order
1. **Exp 4 first** (cheap, ~1–2h): parallel-GPU advantage — a guaranteed ES win to bank.
2. **Exp 1** (the crux): MATH L3-5 3-way — does ES match GRPO OOD when it learns?
3. Then **Exp 2 → Exp 3** (cost + scaling) only if Exp 1 shows ES is OOD-competitive.
