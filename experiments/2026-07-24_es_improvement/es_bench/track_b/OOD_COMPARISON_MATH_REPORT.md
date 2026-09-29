# Cross-method OOD comparison on 3B — trained on MATH L3-5 (proper difficulty) + Countdown OOD

**Bottom line: on Qwen2.5-3B-Instruct, at a 200-step / seed-0 budget, NO method converts training
into held-out gains. GRPO genuinely learns its train reward but does not transfer (held-out MATH-500
and OOD-avg both drop). Full-param ES degrades, monotonically less so as N grows (recovers toward
base at N≥20 but never beats it). LoRA-ES is inert. At large N, ES ≈ GRPO on OOD-avg — both sit
just below base.** This is the MATH-difficulty follow-up to the GSM8K run (where the base was
near-saturated at 0.867 and nobody could improve either).

Model Qwen2.5-3B-Instruct. Train: MATH L3-5 (the ES trainers filter to levels 3-5), 200 steps,
seed 0, binary reward. Eval: full battery, greedy pass@1, cap 300. **Countdown** added as an extra
OOD probe (numbers→equation game; own `<answer>` verifier from `countdown/countdown_task.py`;
prompt = the dataset's own `context`). OOD-avg = mean(svamp, minerva, olympiad, math500, countdown).
KL = base-greedy NLL-drift proxy (×1e3).

| method | MATH500 L3-5 | ΔL3-5 | math500 | gsm8k | svamp | minerva | olymp | amc23 | countdown | OOD-avg | ΔOOD | KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.447 | — | 0.530 | 0.867 | 0.913 | 0.162 | 0.190 | 0.400 | 0.080 | 0.375 | — | — |
| LoRA-ES N=16 | 0.433 | −0.014 | 0.520 | 0.863 | 0.913 | 0.173 | 0.163 | 0.375 | 0.090 | 0.372 | −0.003 | 2.2 |
| **GRPO** | 0.392 | −0.055 | 0.490 | 0.843 | 0.920 | 0.158 | 0.133 | 0.450 | 0.103 | 0.361 | −0.014 | — |
| full-param ES N=4 | 0.235 | −0.212 | 0.340 | 0.730 | 0.827 | 0.077 | 0.090 | 0.100 | 0.043 | 0.275 | −0.100 | 137.1 |
| full-param ES N=8 | 0.332 | −0.115 | 0.430 | 0.857 | 0.887 | 0.129 | 0.120 | 0.300 | 0.030 | 0.319 | −0.056 | 54.2 |
| full-param ES N=20 | 0.429 | −0.018 | 0.503 | 0.850 | 0.883 | 0.140 | 0.133 | 0.375 | 0.127 | 0.357 | −0.018 | 37.9 |
| full-param ES N=30 | 0.415 | −0.032 | 0.500 | 0.850 | 0.897 | 0.136 | 0.180 | 0.250 | 0.090 | 0.360 | −0.015 | 29.1 |

## Reading (single seed, cap-300 eval → noisy ±~0.03; treat as directional)
- **GRPO learns but doesn't transfer.** Train reward rose 0.24→0.59 over 200 steps and responses
  shortened 489→345 tokens — it clearly optimized the objective — yet held-out MATH-500 L3-5 *fell*
  (0.447→0.392) and OOD-avg fell −0.014. The only OOD set it helped was countdown (+0.023). Classic
  train-reward-up / held-out-down (mild reward shaping toward shorter, higher-scoring completions).
- **Full-param ES degrades, monotone in N.** N4 −0.212 (model destroyed, KL 137) → N8 −0.115 →
  N20 −0.018 → N30 −0.032; KL falls 137→29e-3. Bigger population = cleaner gradient = less drift,
  recovering *toward* base at N≥20 but never beating it. Identical shape to the GSM8K run.
- **LoRA-ES inert.** KL≈0.002, held-out flat. (Its own MATH-L3-5 val ID moved 0.34→0.40, but that
  in-distribution wiggle does not appear on the held-out MATH-500 benchmark.)
- **ES ≈ GRPO at large N.** OOD-avg: GRPO 0.361, ES N20 0.357, ES N30 0.360 — statistically
  indistinguishable, all ~0.015 below base. The "closest to base" arm is LoRA-ES (0.372) purely by
  not moving. No arm shows positive held-out transfer.
- **Countdown as an OOD probe worked as intended:** base solves only 0.08 (format followed 92% of
  the time — solving, not formatting, is the bottleneck), so it has real headroom. The near-base
  methods nudged it up (ES N20 0.127, GRPO 0.103, LoRA 0.090); heavy-drift small-N ES killed it
  (N4/N8 0.03-0.04) as format degrades under large weight drift.

## Conclusion
Moving from near-saturated GSM8K to proper-difficulty MATH L3-5 did **not** surface a learning→OOD
win for ES on 3B at this budget — and GRPO also fails to convert its (real) training gains into
held-out improvement. Against the paper's thesis (ES ≈ GRPO on OOD while saving VRAM): here ES ≈ GRPO
holds only in the trivial sense that *both fail to beat base* at large N, and ES's LoRA speed/VRAM
win (A0: 2.7-3.6× faster/step, lowest VRAM) remains unrealized without a positive learning signal.

## Caveats / next
- Single seed, 200 steps, cap-300 eval. The GRPO non-transfer and the ES N20-vs-N30 ordering are
  within noise — need seeds {1,2} and/or larger eval cap to firm up.
- GRPO trained on the `data_math` mix; consider matching its train filter exactly to ES's L3-5.
- ES learned on MATH L3-5 in Phase 4 (1.5B, different setup) — the 3B degradation suggests the
  200-step/σ budget is mis-tuned for 3B more than a task-learnability failure.

## Deliverables
results/exp3b_math/{base_3b,grpo_full_3b,fulles_N{4,8,20,30}_summary,loraes_N16_summary}.json ·
results/exp3b_math/OOD_COMPARISON_MATH.csv · per-question dumps `*_finaleval__<set>_perq.jsonl`
(incl. `__countdown_perq.jsonl`). Countdown wired into `track_{a,b}/src/eval_core.py` (`_eval_countdown`,
default-on via `include_countdown`). Driver: `track_b/run_exp3b_math.sh`.
