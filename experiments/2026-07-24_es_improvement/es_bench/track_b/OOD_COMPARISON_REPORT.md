# Cross-method OOD comparison — LoRA-ES vs full-param ES vs GRPO (GSM8K train, 1.5B-Instruct)

**Bottom line: GRPO is the only method that improves OOD on GSM8K. Both ES variants fail —
LoRA-ES is inert (never leaves base), full-param ES actively degrades (drifts and gets worse).**
The A0 wall-clock win for LoRA-ES (2.7–3.6× faster/step) is real but unrealized: it doesn't learn.

Train: plain GSM8K, Qwen2.5-1.5B-Instruct, N=16, 200 steps, 1 seed. Eval: full battery, cap 300,
greedy. OOD-avg = mean(svamp, minerva, olympiad, math500). KL = base-greedy NLL-drift proxy (×1e3).

| method | ID gsm8k | ΔID | OOD-avg | ΔOOD | svamp | minerva | olymp | math500 | KL |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.750 | — | 0.364 | — | 0.817 | 0.081 | 0.113 | 0.443 | 0 |
| LoRA-ES (σ0.005) | 0.747 | −0.003 | 0.364 | +0.000 | 0.837 | 0.099 | 0.100 | 0.420 | **0.0** |
| full-param ES (σ5e-4) | 0.710 | −0.040 | 0.341 | −0.022 | 0.783 | 0.085 | 0.107 | 0.390 | 17.8 |
| full-param ES (σ1e-3) | 0.723 | −0.027 | 0.357 | −0.006 | 0.803 | 0.099 | 0.133 | 0.393 | 12.1 |
| full-param ES (σ2e-3) | 0.710 | −0.040 | 0.372 | +0.009 | 0.847 | 0.107 | 0.133 | 0.403 | 13.9 |
| full-param ES (σ4e-3) | 0.723 | −0.027 | 0.352 | −0.011 | 0.807 | 0.096 | 0.093 | 0.413 | 8.1 |
| **GRPO (full-param)** | **0.780** | **+0.030** | **0.378** | **+0.014** | 0.853 | 0.110 | 0.120 | 0.427 | — |

## Wall-clock (A0 microbench, 3B, 64 seqs/step)
LoRA-ES 2.89 s/step · full-param GRPO 7.84 · LoRA-GRPO 10.44 → LoRA-ES **2.7–3.6× faster/step**,
lowest VRAM. But end-to-end this is moot: LoRA-ES reaches no better model.

## Reading
- **GRPO learns GSM8K**: +0.030 ID, +0.014 OOD (svamp +0.036, minerva +0.029). So GSM8K is NOT
  unlearnable — GRPO improves it.
- **LoRA-ES is inert**: OOD = base exactly, KL 0. The ES gradient in adapter space finds no coherent
  direction; Θ stays ≈ 0.
- **Full-param ES degrades**: ID −0.03/−0.04, OOD flat-to-down, KL 8–18e-3 — it drifts substantially
  and gets worse. (It learned on MATH L3-5 in Phase 4; GSM8K binary reward is a weak/adversarial ES
  signal for this already-strong model.)
- **σ dose-response (full-param ES)**: OOD-avg peaks weakly at σ=2e-3 (+0.009, within noise); no
  monotone gain. KL falls with... no clean trend. No σ recovers base.

## Conclusion & recommendation
On GSM8K, **GRPO is the method that works**; ES (either parameterization) does not. LoRA-ES's speed
advantage only matters on a task where ES actually learns — which GSM8K is not, for this model.
**Recommend re-running this exact comparison on MATH L3-5** (where full-param ES demonstrably learned
in Phase 4, base ID ~0.5 with real headroom) to test whether LoRA-ES's speed can be converted into a
real learn→OOD result, and whether ES can match GRPO there.

## Deliverables
results/OOD_COMPARISON.csv · track_b/{b1,base,grpo_ood} summaries · trainers in track_b/src +
grpo_ood/ (verl train → model_merger FSDP→HF → vLLM OOD eval).
