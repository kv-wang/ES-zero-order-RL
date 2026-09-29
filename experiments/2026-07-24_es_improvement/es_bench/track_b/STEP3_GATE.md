# Step 3 — decisive LoRA-ES re-tune gate

Model Qwen2.5-1.5B-Instruct, MATH L3-5, N=16, B=8, 300 steps, seed 0, greedy fitness, vanilla
z-score shaping. Single-variable discipline: only (σ, α) move. Driver `run_step3_decisive.sh`;
raw logs `results/step3_decisive/*.jsonl`, `logs/step3_decisive/`.

**Pre-registered gate:** *train fitness rises clearly beyond the control* **AND** *KL ∈ [1e-3, 3e-2]*.

**Verdict: GATE NOT MET.** Neither condition is satisfied by any arm. Detail below, including a
power analysis showing the fitness condition was **not measurable** at B=8 as specified.

---

## Run status

| arm | σ | α | steps | status |
|---|---|---|---|---|
| control | 5.0e-3 | 2e-3 | 300/300 | complete (1708 s) |
| B | 7.7e-3 | 3e-3 | 300/300 | complete (1711 s) |
| A | 7.7e-3 | 1e-2 | **128/300** | **terminated early — session loss at 20:31, process killed mid-run** |

Arm A was **not** run to completion and no summary JSON exists for it. It is adjudicated on its
128-step partial trajectory; the justification for not re-running is in §4. Everything reported for
A is labelled as partial.

## 1. Results

| arm | ID @0 | ID @300 | ΔID | KL final | KL/step | zero-update | ‖Δ‖ (non-zero) |
|---|---|---|---|---|---|---|---|
| control | 0.355 | 0.335 | −0.020 | 9.05e-3 | 3.02e-5 | 1.3% | 2.149 |
| B | 0.355 | 0.365 | **+0.010** | **3.10e-2** | 1.03e-4 | 0.7% | 3.223 |
| A (partial) | 0.355 | **0.050 @100** | −0.305 @100 | **0.324 @100** | 3.2e-3 | 31% over [100,128) | 10.740 |

ID is measured on a 200-problem held-out set, so its binomial SE at p≈0.35 is **±0.034**. Both
ΔID = −0.020 and ΔID = +0.010 are well inside one SE of zero. **Neither arm moved held-out accuracy.**

ID trajectory for B is visibly noise-dominated, not a trend: 0.355 → 0.345 → 0.325 → 0.330 → 0.345
→ 0.370 → 0.365.

## 2. Condition 1 — train fitness vs control: **FAILS**

Two estimators on the same logs. `f20−l20` is the first-20 vs last-20-step window difference (the
estimator the +0.022 threshold came from); OLS is the least-squares slope over all steps, scaled to
the full run.

| arm | n | sd(fit) | f20−l20 | SE | t | OLS rise | t |
|---|---|---|---|---|---|---|---|
| control | 300 | 0.142 | −0.0242 | 0.0449 | −0.54 | +0.0334 | +1.18 |
| B | 300 | 0.136 | −0.0348 | 0.0432 | −0.81 | +0.0255 | +0.94 |
| A (partial) | 129 | 0.143 | −0.3145 | 0.0453 | −6.95 | −0.3752 | −13.41 |

Neither control nor B shows a fitness rise distinguishable from zero (|t| ≤ 1.2 on either
estimator). The two estimators even disagree in **sign** for control and B — the window estimator
says down, the slope says up, and both are inside noise.

**Paired test (the sharpest available).** Both arms share CRN — identical prompt batches at every
step — so per-step differences can be paired:

| window | mean(B − control) | SE | t |
|---|---|---|---|
| all 300 steps | **−0.0121** | 0.0030 | **−4.00** |
| last 100 steps | −0.0100 | 0.0067 | −1.50 |

B is not above control; over the full run it is **significantly below** it. The gate asked for B to
rise *clearly beyond* control and it did the opposite.

## 3. Condition 2 — KL band [1e-3, 3e-2]: **control passes, B misses, A catastrophic**

- control 9.05e-3 — inside.
- B **3.10e-2 — above the ceiling**, by 1.03×. Marginal, but it is outside as pre-registered.
- A 0.324 by step 100 — **10.8× the ceiling**, still climbing.

Since the gate is a conjunction, control passing on KL is irrelevant: control is the before-fix
baseline and fails condition 1 by construction.

## 4. Why arm A is adjudicated rather than re-run

A's purpose was to bracket the Step-2 / Step-3 gate conflict — the velocity gate chose α=1e-2, the
KL band implied α≈2.5e-3. The 128 steps collected settle it:

| step | ID | KL | vs band ceiling |
|---|---|---|---|
| 0 | 0.355 | ~0 | — |
| 50 | 0.215 | 7.09e-2 | 2.4× over |
| 100 | 0.050 | 3.24e-1 | 10.8× over |

The KL band was already breached by step 50 and the model is destroyed by step 100 (ID 0.050 —
below the 0.355 base by 9 SE, and near the floor where extraction itself fails). Fitness slope is
−0.375 per 300 steps at t = −13.4, and zero-update steps appear at 31% over [100, 128) — the
freeze mechanism from D2 Step 2 (adapter destroyed → all members score 0 → std 0 → no update),
already documented at α ≥ 5e-2 and here reached at 1e-2 given 128 steps.

**The D2 extrapolation is confirmed.** Step 2 fitted p=1.56 and predicted KL(300) ≈ 1.4 for α=1e-2.
Observed KL(100) = 0.324; continuing the same power law from that point gives KL(300) ≈ 1.8 — same
order. Running the remaining 172 steps would refine a number for an arm already 10× outside its own
gate, at ~16 min GPU, and cannot change any verdict. **The Step-2 velocity criterion is rejected;
the Step-3 KL band was the correct selector.** Recorded as a deliberate early stop, not a silent
truncation.

## 5. The finding that matters most — the gate's fitness criterion was unmeasurable

The +0.022 threshold was borrowed from the 3B LoRA N=16 arm (Phase-0 audit, Table 6). Tested
against its own noise:

| arm (3B MATH, warmup excluded) | n | f20−l20 | SE | t | OLS rise | t |
|---|---|---|---|---|---|---|
| **LoRA-ES N=16** | 200 | **+0.0223** | 0.0476 | **+0.47** | +0.0091 | +0.25 |
| full-param N=4 | 195 | −0.0906 | 0.0429 | −2.11 | −0.0848 | −2.57 |
| full-param N=8 | 195 | +0.0312 | 0.0455 | +0.69 | +0.0314 | +0.88 |
| full-param N=20 | 195 | +0.1016 | 0.0474 | +2.14 | +0.0777 | +2.12 |
| full-param N=30 | 195 | +0.1171 | 0.0476 | +2.46 | +0.0880 | +2.40 |

**The +0.022 threshold is 0.47 SE — indistinguishable from zero.** The gate asked Step 3 to beat a
number that was itself noise. With per-step fitness sd ≈ 0.14 at B=8, a 20-step window mean has
SE ≈ 0.031 and the window *difference* has SE ≈ 0.045; no effect below roughly ±0.09 is detectable
by that estimator at all. This is the direct consequence of the Phase-0 Table 3 quantization floor
(B=8 → 1/8 quantum, sub-quantum population spread on 98% of steps).

Two corrections follow:
- The Phase-0 audit's claim that **LoRA-ES N=16 "raises train fitness (+0.022)"** does not survive
  a significance test and should be withdrawn.
- The same audit's claim for **full-param N=20/N=30 does survive** (t = 2.1 / 2.5), so
  "ES optimizes its training objective at large N and fails to transfer" still stands — for
  full-param at N ≥ 20 only. (My window numbers are larger than the audit's because warmup steps
  are excluded here; the LoRA figure is identical either way.)

## 6. What D2 + Step 3 jointly establish

1. **σ was never the problem** — σ\* = 7.7e-3 vs the 5e-3 already in use (1.5×). Confirmed by Step 3:
   moving σ to σ\* changed nothing that matters (ΔID +0.010, inside noise; fitness *below* control).
2. **α is the live axis, and its usable knee is below 1e-2** — α=1e-2 destroys the adapter within
   100 steps; α=3e-3 is inert; α=2e-3 is inert. The window between "does nothing" and "destroys"
   contains no setting that learns.
3. **Defect A is real but is not the cause of inertness.** The broken α/σ coupling meant A1's sweep
   was never a step-size sweep — that critique stands. But repairing the calibration and sweeping
   the axis that was never varied still yields no learning. **A1's negative was void as reasoned,
   yet correct as concluded.**
4. **The B=8 fitness signal cannot support this class of gate.** Any future criterion phrased on
   train-fitness deltas needs either larger B or many seeds; at B=8/1 seed the measurement floor is
   ±0.09.

Context: the GRPO 3B MATH checkpoint measures KL 0.129 on this same proxy — an order of magnitude
above every non-destroyed ES arm here — while also failing to transfer (MATH500 L3-5 0.447→0.392).
Large KL is not what ES is missing.

## 7. Options

- **(a) Accept the negative and write it up.** The LoRA-ES inertness result now has a properly
  calibrated σ, a swept α, and a mechanistic account (constant-norm update + sub-quantum fitness
  spread). This is a publishable negative with the confound Phase-0 identified now closed.
- **(b) Attack the measurement floor first — Phase 2 as originally scoped.** Raise B (64/128) so
  fitness spread exceeds shot noise, then re-ask whether α has a working window. Phase-0 costed
  this at ~117 h single-GPU worst case and flagged that a 10-step microbench must precede it.
  This is the only option that tests the leading mechanistic hypothesis.
- **(c) Seeds before anything else.** Everything above is seed 0. The 3-seed rule means no
  conclusion here is final; 2 more seeds on control + B is ~1.9 h and would firm up §2.
- **(d) Drop the estimator, not the program.** Re-run the gate on held-out ID with a paired
  per-question test rather than train fitness — cheaper than (b) and measures what we actually care
  about.

**Recommendation: (c) then (a).** Two more seeds is under two hours and is required by the
program's own 3-seed rule before any negative is stated; (b) is a large spend against a hypothesis
that Step 3 has already made less likely, since the α window is bounded by adapter destruction
rather than by signal quality.

**STOP — awaiting direction.**
