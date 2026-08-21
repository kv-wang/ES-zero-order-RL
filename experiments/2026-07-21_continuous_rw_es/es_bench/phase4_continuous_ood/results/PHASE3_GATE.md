# Phase 4 — Phase 3 gate (seeds, KL): the pre-registered read-out

16 ES arms, 200 steps, in-process eval (cap 300) + KL proxy. seeds: N∈{2,4} {0,1,2};
N=30 {1,2} (+Phase-2 s0 accuracy). KL = base-greedy NLL-drift proxy; ratio gate valid.
**GRPO control still pending.**

## Per-arm (mean ± sd over seeds)
| arm | ID L3-5 | OOD-avg | KL ×1e3 | zero-upd | pure-tb |
|---|---|---|---|---|---|
| bin N2 | 0.498±0.011 | 0.528±0.006 | 13.11 | 0.563 | 0.000 |
| con N2 | 0.462±0.009 | 0.508±0.003 | **34.11** | 0.000 | 0.553 |
| bin N4 | 0.504±0.017 | 0.528±0.001 | 10.51 | 0.293 | 0.000 |
| con N4 | 0.482±0.026 | 0.531±0.004 | 14.69 | 0.000 | 0.283 |
| bin N30 | 0.508±0.021 | 0.543±0.006 | 1.49 | 0.067 | 0.000 |
| con N30 | 0.513±0.004 | 0.545±0.006 | 1.86 | 0.000 | 0.053 |

## Within-N contrast (continuous − binary) vs pre-registered gates
| N | ΔOOD | 2SD | ΔID | KL ratio | **verdict** |
|---|---|---|---|---|---|
| 2 | **−0.020** | 0.009 | **−0.035** | **2.60** | **NEGATIVE** |
| 4 | +0.003 | 0.006 | −0.022 | 1.40 | NEUTRAL |
| 30 | +0.003 | 0.011 | +0.005 | 1.25 | NEUTRAL |

- **GO condition met at NO N.** The pre-registered GO (cont OOD ≥ binary +2SD AND KL ≤ 1.2×) fails everywhere.
- **N=2 is a clean NEGATIVE**: continuous significantly hurts OOD (−0.020 > 2SD) *and* ID (−0.035 > 2SD) *and* sharpens KL 2.6× (> the 1.5× negative bound).
- **N=4, N=30 NEUTRAL**: no OOD effect within noise; ID trends slightly negative at N=4; KL elevated but sub-1.5×.

## Diagnostic: tiebreaker dosage drives sharpening (pre-registered prediction — CONFIRMED)
Pure-tiebreaker fraction and excess KL (cont − bin) both fall monotonically with N:
| N | pure-tb frac | excess KL ×1e3 | KL ratio |
|---|---|---|---|
| 2 | 0.553 | 21.0 | 2.60 |
| 4 | 0.283 | 4.2 | 1.40 |
| 30 | 0.053 | 0.4 | 1.25 |
The more tie-steps the secondary "rescues" (small N), the more the model sharpens toward
gold tokens and drifts from base — exactly the noise-as-regularizer mechanism. The rescued
steps trade a free zero-update for excess KL that does **not** buy OOD generalization.

## Bottom line
**Hypothesis refuted.** A continuous lexicographic tiebreaker does not improve ES OOD at
small N; at the smallest N=2 it actively harms it via likelihood-sharpening. This is a clean,
mechanistically-explained NEGATIVE — evidence *for* the noise-as-regularizer view. Binary
zero-update rates reproduced Q1 across all seeds, and the mechanism did exactly what it was
built to do (kill dead steps) — it simply isn't beneficial.

## Remaining
- GRPO seeds {0,1,2} (control for the "vs GRPO" section) — GPUs now free; needs verl ckpt→HF→vLLM eval.
- REPORT.md (final): tables + plots (OOD vs N by reward; KL vs N; pure-tb & zero-upd vs N), confound list, verbatim GRPO-binary-reward limitation.
- No design change: per pre-registration, a clean negative is a valid outcome — do NOT tune to rescue.
