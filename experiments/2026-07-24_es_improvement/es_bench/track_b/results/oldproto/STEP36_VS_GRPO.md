# LoRA-ES B=200 step-36 vs GRPO — same protocol (2026-07-28)

**Why this run.** The overnight B=200 MATH arm reported ΔOOD +0.0155 and was called the program's
first positive transfer. Its eval used `max_tokens=512`; every earlier 3B number (base, GRPO,
full-param ES) used `max_tokens=2048` (`eval_base.py:23`, `eval_grpo_ood.py:46`). Decomposing the
512-token result showed the entire gain was extract-rate (+0.056..+0.088) with accuracy-given-
extraction flat (+0.001/−0.026/−0.006), i.e. the adapter learned to finish inside the cap.
This run re-evaluates the same `theta_step36_final.pt` at cap-300 / 2048 tokens.

**Validity check: the base row reproduced `base_3b.json` exactly on all 7 datasets** (0.5300,
0.8667, 0.9133, 0.1618, 0.1900, 0.4000, 0.0800), so the protocol is deterministic and the
previously-computed GRPO row is directly comparable without re-running it.

## Results (Qwen2.5-3B-Instruct, greedy, cap 300, max_tokens 2048)

| method | MATH500 | L3-5 (ID) | gsm8k | svamp | minerva | olympiad | amc23 | countdown | OOD-avg |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.5300 | **0.4470** | 0.8667 | 0.9133 | 0.1618 | 0.1900 | 0.400 | 0.0800 | **0.3750** |
| LoRA-ES B=200 step36 (λ=1) | 0.5267 | **0.4470** | 0.8833 | 0.9167 | 0.1544 | 0.1767 | 0.275 | 0.0667 | 0.3682 |
| LoRA-ES B=200 step36 (λ=0.5) | 0.5367 | 0.4608 | 0.8833 | 0.9167 | 0.1618 | 0.1667 | 0.400 | 0.0833 | 0.3730 |
| GRPO full-param 200 steps | 0.4900 | 0.3917 | 0.8433 | 0.9200 | 0.1581 | 0.1333 | 0.450 | 0.1033 | 0.3609 |
| LoRA-ES N16 B=8 (old) | 0.5200 | 0.4332 | 0.8633 | 0.9133 | 0.1728 | 0.1633 | 0.375 | 0.0900 | 0.3719 |
| full-param ES N20 | 0.5033 | 0.4286 | 0.8500 | 0.8833 | 0.1397 | 0.1333 | 0.375 | 0.1267 | 0.3573 |
| full-param ES N30 | 0.5000 | 0.4147 | 0.8500 | 0.8967 | 0.1360 | 0.1800 | 0.250 | 0.0900 | 0.3605 |

OOD-avg = mean(math500, svamp, minerva, olympiad, countdown), the definition used in
`exp3b_math/OOD_COMPARISON_MATH.csv`. SEs: math500/olympiad ≈0.029/0.023, minerva 0.022,
countdown 0.016, gsm8k 0.020, amc23 **0.078** (n=40 — its ±0.125 swings are noise).

## Findings

1. **The +0.0155 OOD gain does not survive the protocol change.** At 2048 tokens the step-36
   adapter is −0.0068 OOD vs base, and its ID primary metric (MATH500 L3-5) is **0.4470 —
   identical to base to four digits**. Every per-dataset delta is inside 1.6 SE.
2. **The mechanism is confirmed as truncation relief.** At 512 tokens the adapter's extract rate
   rose +0.056/+0.070/+0.088 on math500/minerva/olympiad. At 2048 those same deltas are
   +0.010/+0.037/−0.017 — the effect disappears once the cap stops binding, which is what a
   length/format effect predicts and a capability gain does not.
3. **GRPO is the worst arm on this battery**: OOD-avg 0.3609 (−0.0141), L3-5 0.3917 (−0.055),
   olympiad −0.0567 (2.5 SE). It trains (reward 0.24→0.59) and does not transfer.
4. **Nobody beats base.** Ranked by OOD-avg: base 0.3750 > LoRA-ES-λ0.5 0.3730 > LoRA-ES N16 B=8
   0.3719 > **LoRA-ES B=200 step36 0.3682** > full-ES N30 0.3605 ≈ GRPO 0.3609 > full-ES N20
   0.3573. The ordering is "how little did you move", as in every previous phase.

## Consequence for the program

Finding #3 of `OVERNIGHT_FINDINGS.md` ("first positive transfer") is **withdrawn**. Findings #1
(shot-noise floor gone at B=200: split-half ρ +0.46, 0 sub-quantum steps) and #2 (LoRA-ES speed
advantage reverses at B=200) are unaffected — they are measured on training internals and
wall-clock, not on this eval.

The seed-confirmation run is no longer the obvious next step: it would confirm a result that has
now been explained away. The open question B=200 raised is narrower — the estimator finally has a
real ranking signal (ρ +0.46), but 36 steps of it produced zero measurable capability change at
KL 8.8e-3.

Artifacts: `results/oldproto/loraes_math_step36.json`, `logs/oldproto/loraes_step36.log`,
runner `src/eval_oldproto.py`.
