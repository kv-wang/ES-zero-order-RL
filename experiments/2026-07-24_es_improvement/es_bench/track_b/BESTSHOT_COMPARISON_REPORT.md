# Best-shot LoRA-ES vs GRPO at matched wall-clock — 3B, MATH L3-5 (2026-07-29)

**Question.** Given its fairest measured configuration and exactly GRPO's wall-clock, can LoRA-ES
match GRPO's OOD accuracy on Qwen2.5-3B-Instruct? (User goal: demonstrate an ES benefit — either
similar OOD at less time, or better OOD at similar time.)

**This is the first fully matched cross-method row in the program**: same train budget (GRPO's own
summed step times, 5178 s), same train cap (2048 = GRPO's `max_response_length`), same eval
protocol (cap 300, max_tokens 2048, greedy), same engine (vLLM 0.11, `gpu_memory_utilization`
0.50 both methods per the 2026-07-28 config lock), same data (MATH L3-5, `data_math` split),
single GPU (H200 NVL), seed 0, fully unattended (`run_grpo_chain.sh` + `run_es_bestshot_chain.sh`).

## Arms

- **GRPO** full-param, verl 0.8: 200 steps × 25.89 s/step, bs 16 × 8 rollouts, lr 1e-6, KL coef
  1e-3. Train reward 0.601 → 0.676 (real optimization); resp len stable ~682; clip ratio 1% (no
  truncation-driven brevity learning — the @512 pathology is gone at 2048).
- **LoRA-ES best-shot**: every lever from `LORAES_WALLCLOCK_ANALYSIS.md` — N=30, **B=100**
  (fidelity/√cost optimum), **attn-only rank 8** (d=3.69 M, 8.1× down), **antithetic** (15
  mirrored pairs), **σ\*=0.015 re-selected** under the new d by the Phase A flip-rate procedure
  (flip 0.309 vs reference 0.301; note the r8 destruction knee is sharp — σ=0.03 already flips
  0.895 with KL 0.68), α=σ\*/2=0.0075. Ran **32 steps × 162.6 s/step** to the deadline.

## Results (cap 300 / max_tokens 2048; OODavg over math500·svamp·minerva·olympiad·countdown)

| arm | OODavg | MATH-L3-5 | gsm8k | svamp | minerva | olympiad | countdown | math500 |
|---|---|---|---|---|---|---|---|---|
| **base** | **0.3750** | **0.4470** | 0.8667 | 0.9133 | 0.1618 | 0.1900 | 0.0800 | 0.5300 |
| GRPO@2048 | 0.3721 | 0.4194 | 0.8767 | 0.9200 | 0.1471 | 0.1800 | 0.1033 | 0.5100 |
| LoRA-ES λ=1 | 0.3589 | 0.4147 | 0.8533 | 0.9033 | 0.1581 | 0.1667 | 0.0633 | 0.5033 |
| LoRA-ES λ=0.5 | 0.3636 | 0.4147 | 0.8800 | 0.9067 | 0.1544 | 0.1867 | 0.0767 | 0.4933 |

The base row was produced in the same eval run (row 0 of `eval_oldproto`) and **reproduced
`base_3b.json` exactly** — protocol determinism holds again; all rows are directly comparable.

**Ranking: base ≥ GRPO > LoRA-ES.** Nobody beats base. Per-set deltas are ≤ ~1.6 SE; the ordering
is again "how little did you move". GRPO's +0.075 train-reward gain does not transfer (L3-5
−0.028 vs base). ES λ=1 is −0.016 OODavg / −0.032 L3-5; halving the displacement (λ=0.5) recovers
part of it — the shrinkage-dominates pattern from Phase 1 yet again.

## Why the ES arm didn't learn — the estimator diagnostics

- **fit slope ≈ 0** (−4.7e-4/step; first20 0.554, last20 0.550) and **KL to base 0.0032 at step
  25** — the policy barely moved and not in a useful direction.
- **median split-half ρ = 0.063 at B=100.** This is the decisive number. The §2 cost table
  predicted ρ≈0.32 at B′=100 by subsampling the B=200 bitmaps; the fresh arm measured **5× less
  ranking signal** than that extrapolation. The subsampling analysis conditioned on the B=200
  run's members and its (σ, r16 attn+mlp, non-antithetic, cap-512) regime; it did not survive the
  transfer to the r8/antithetic/cap-2048 regime. With ρ≈0.06 the per-step update direction is
  mostly noise, and 32 steps of noise is what the eval shows.
- Gates never fired: the KL guard (3e-2) was never approached, and the step-60 slope/ρ gate was
  past the 32-step horizon. For future short arms, scale `--gate_step` to the expected step count.

## Wall-clock and memory

- ES per-step cost is **6.3× GRPO's** (162.6 vs 25.9 s) at matched engine settings. 200 GRPO
  steps ≙ 32 ES steps. (§6's corrected 24–42-step forecast: measured 32. The forecast holds.)
- GRPO peak VRAM this run: **106.7 GB** (vLLM 0.50 + resident FSDP actor + optimizer). LoRA-ES
  was not resampled this run; it is engine-bound at util 0.50 (~72 GB) with no training state —
  the ~35 GB VRAM margin remains ES's one demonstrated advantage.

## Verdict against the user's two success criteria

1. *"Similar OOD accuracy while saving wall-clock"* — **not achieved**: at equal wall-clock ES is
   below GRPO and both are below base; and per unit of learning, ES is the slower method here,
   not the faster one (its A0 speed win was a small-batch artifact, retracted 2026-07-28).
2. *"Better OOD at similar wall-clock"* — **not achieved**: ES λ=1 is the worst row in the table.

The honest structural read stands (§6 of the wall-clock analysis): GRPO extracts a full gradient
from 128 rollouts/step; ES extracts 30 scalars from 3,000 — and at this budget the 47× per-rollout
information gap is not closed by B-tuning, d-reduction, antithetic, or σ calibration combined.
The demonstrable ES benefits on this task/model/budget are **VRAM (~35 GB)** and **simplicity (no
backward pass)** — not wall-clock, not OOD accuracy.

## Caveats and what could still change the picture

- **Single seed** (seed 0). The 3-seed rule applies before any paper claim; but note every
  multi-seed and multi-config ES result in this program to date points the same direction.
- The ρ collapse at B=100-fresh vs B=200-subsampled is **new information**: if anything rescues
  ES here it is larger B (ρ was 0.46 at B=200), i.e. *fewer, better steps* — the opposite of the
  matched-clock regime, which forces many weak steps. A B=200 arm at this budget would get ~16
  steps. That trade has no measured winner yet; it is the one open experimental question this
  campaign leaves.
- Base being unbeatable by *either* method suggests 3B-at-this-budget has no OOD headroom for
  RL-style post-training on MATH — consistent with every 3B result since 2026-07-26. Showing an
  ES benefit may require a setting where *something* beats base first.

---

# Addendum (2026-07-29): "why is everyone below base?" — paired diagnosis

The aggregate table invites a wrong reading. Three paired tests (`run_ood_diagnosis.sh` →
`results/ood_diag/DIAGNOSIS.json`; McNemar on identical problems, ES λ-grid dose-response,
train-pool-vs-held-out memorization probe) give the two methods **different verdicts**:

**GRPO is NOT below base.** Paired OODavg delta −0.0029 ± 0.0078 (z = −0.38). The aggregate
−0.003 was noise, and the report's "base ≥ GRPO" ordering should not be read as an ordering.
What GRPO actually did: it flips a lot of answers (38/300 changed on math500) with wins ≈ losses
— a sideways move. And the memorization suspicion was **wrong**: on its own 300 train-pool
problems GRPO gains only +0.017 (z = 0.7) while on 300 *fresh same-distribution* problems it
gains +0.033 (z = 1.4); diff-in-diff −0.017 ± 0.034 ⇒ **no memorization signal at all**. GRPO
learned a small, genuine, on-distribution improvement that nets to ~zero on the OOD battery
(mild positive on countdown +0.023, mild negative on math500 −0.02, both ns).

**The ES deficit IS real — and it is drift damage, mechanism confirmed.** Paired OODavg delta at
λ=1: **−0.0155 ± 0.0072 (z = −2.16)**, negative on all 5 sets (no single-set artifact). The
dose-response runs 0.3750 (base) → 0.3709 (λ.25) → 0.3636 (λ.5) / 0.3695 (λ.75) → 0.3596 (λ1) —
damage grows with displacement and shrinkage recovers it (λ.25 and λ.75 are statistically at
base, z −0.69/−0.80; the λ.5/.75 inversion is inside one SE). Combined with the training-time
diagnostics (split-half ρ = 0.06 ⇒ ~94%-noise update direction; KL 0.003), the causal chain is
closed: *noise-dominated steps ⇒ random-walk displacement ⇒ uniform small OOD damage that scales
away linearly with λ* — the same signature Phase 1 measured for full-param ES.

**ES vs GRPO directly:** paired delta −0.0125 ± 0.0078 (z = −1.61) — suggestive, not significant.

**Corrected reading of the main table:**

| claim | paired verdict |
|---|---|
| GRPO < base | **not supported** (z −0.38); GRPO ≈ base on OOD, small real on-distribution gain |
| ES λ1 < base | **confirmed** (z −2.16), all sets; pure drift damage, recovered by shrinkage |
| GRPO > ES | suggestive only (z −1.61) |

**What "resolving" it means.** For GRPO there is nothing to resolve — the deficit doesn't exist;
the accurate statement is "GRPO neither gains nor loses OOD at this budget." For ES the deficit
is real but fully mechanistic: it is the cost of taking 32 steps whose direction was 94% noise.
It cannot be tuned away at fixed signal (α smaller ⇒ same walk, slower; λ-shrinkage ⇒ returns to
base, i.e. undoes the training). The only real fix is raising the per-step signal — B=200 restored
split-half ρ to 0.46 — which at matched wall-clock means ~16 steps. That arm remains the open
question. A KL- or norm-anchored ES (bounding displacement during training rather than shrinking
after) would cap the damage but cannot create a gain that isn't there.

Diagnosis artifacts: `results/ood_diag/{DIAGNOSIS.json,es_eval.json,grpo_eval.json,`
`overfit_base.json,overfit_grpo.json,*_perq.jsonl}`; code `src/{analyze_ood_diagnosis.py,`
`grpo_overfit_probe.py}`, `run_ood_diagnosis.sh`; `eval_oldproto.py --perq` added.

**Artifacts.** `results/grpo_full_3b_math_t2048/{summary.json,ood_eval.json}`,
`results/loraes_bestshot_3b_math/{COMPARISON.json,ood_eval.json,arm.jsonl,arm_summary.json,`
`theta_final.pt,phaseA_r8/SIGMA_STAR.json,timing_5step_summary.json,STATUS.txt}`.
Drivers: `run_grpo_chain.sh`, `run_es_bestshot_chain.sh`.
