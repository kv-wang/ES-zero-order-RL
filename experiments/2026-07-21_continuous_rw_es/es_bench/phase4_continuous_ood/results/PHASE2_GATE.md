# Phase 4 — Phase 2 gate (seed-0 screen, 10 ES arms, 200 steps, in-process eval cap 300)

Wall-clock ~4h03m on 2×H200. All arms rc=0. **GRPO seed-0 still pending** (separate verl
pipeline + checkpoint-eval plumbing). **KL not yet computed** (harness deferred; drift bounded).
Single seed → no SD yet, so the pre-registered ±2SD gates cannot be applied; this screen only
decides which arms proceed to seeds 1–2.

## Training diagnostics — mechanism validated
| N | bin zero-upd | Q1 s0 | con zero-upd | con pure-tiebreak | s/step (b/c) | score% |
|---|---|---|---|---|---|---|
| 2 | 0.520 | — | 0.000 | 0.525 | 2.70/2.86 | 5.1 |
| 4 | 0.325 | **0.325** | 0.000 | 0.310 | 5.38/5.70 | 5.1 |
| 6 | 0.190 | — | 0.000 | 0.220 | 8.07/8.51 | 5.0 |
| 8 | 0.165 | **0.165** | 0.000 | 0.155 | 10.84/11.39 | 5.2 |
| 30 | 0.040 | **0.040** | 0.000 | 0.050 | 40.23/42.42 | 5.1 |

- Binary zero-update **exactly reproduces certified Q1 seed-0** (0.325/0.165/0.040) — the re-run is faithful.
- Continuous **eliminates zero-update at every N** (0.000), and its **pure-tiebreaker rate ≈ the binary zero-update rate** — the secondary fires on exactly the would-be-dead steps, at the ~1/N cadence. Scoring overhead ~5%.

## Final accuracy (in-process, cap 300, seed 0)
| arm | ID L3-5 | svamp | gsm8k | minerva | olympiad | amc23 | OOD-avg |
|---|---|---|---|---|---|---|---|
| bin N2 | 0.511 | 0.900 | 0.867 | 0.143 | 0.207 | 0.575 | 0.529 |
| con N2 | 0.475 | 0.900 | 0.820 | 0.110 | 0.210 | 0.475 | 0.510 |
| bin N4 | 0.516 | 0.907 | 0.857 | 0.147 | 0.197 | 0.450 | 0.527 |
| con N4 | 0.447 | 0.903 | 0.867 | 0.140 | 0.227 | 0.500 | 0.534 |
| bin N6 | 0.507 | 0.910 | 0.857 | 0.154 | 0.207 | 0.550 | 0.532 |
| con N6 | 0.498 | 0.923 | 0.850 | 0.154 | 0.227 | 0.525 | 0.539 |
| bin N8 | 0.488 | 0.913 | 0.870 | 0.158 | 0.217 | 0.550 | 0.540 |
| con N8 | 0.470 | 0.907 | 0.877 | 0.151 | 0.227 | 0.500 | 0.540 |
| bin N30 | 0.479 | 0.923 | 0.880 | 0.169 | 0.207 | 0.525 | 0.545 |
| con N30 | 0.507 | 0.917 | 0.870 | 0.140 | 0.253 | 0.575 | 0.545 |

## Within-N contrast (continuous − binary), seed 0
| N | ΔOOD-avg | ΔID L3-5 | Δ(svamp, gsm8k, minerva, olympiad) |
|---|---|---|---|
| 2 | −0.019 | −0.037 | +0.000, −0.047, −0.033, +0.003 |
| 4 | +0.007 | −0.069 | −0.003, +0.010, −0.007, +0.030 |
| 6 | +0.007 | −0.009 | +0.013, −0.007, +0.000, +0.020 |
| 8 | +0.001 | −0.019 | −0.007, +0.007, −0.007, +0.010 |
| 30 | +0.000 | +0.028 | −0.007, −0.010, −0.029, +0.047 |

## Read (seed-0, provisional)
- **OOD: null.** ΔOOD-avg ∈ [−0.019, +0.007] across all N — within cap-300 eval noise (~±0.02–0.03). No continuous-reward OOD advantage at any N, including the small-N target {2,4}.
- **ID: slightly negative at small N** (ΔID −0.037 at N2, −0.069 at N4), consistent with the pre-registered **noise-as-regularizer / likelihood-sharpening** risk (tie steps most frequent at small N). Only olympiad shows a weak, consistent continuous + (up to +0.047).
- The mechanism does exactly what it was designed to (kills dead steps), but that does **not** translate into OOD gains here → trending toward the pre-registered **NEUTRAL/NEGATIVE** outcome, pending seeds 1–2 for SD and the KL bound.

## Confounds / caveats
- Single seed, cap-300 eval → differences within noise; needs seeds {0,1,2}.
- **Secondary-key saturation** (many P̄=1.0) plausibly blunts the tiebreaker's signal — the rescued steps get a low-information direction.
- Minerva/olympiad extraction (0.55/0.72) noise on the hard end.
- **KL not computed** — required for the final GO/NEUTRAL/NEGATIVE classification (bound 1.2×/1.5×).

## Recommendation
- **No compelling N4↔N8 non-monotonicity** (ΔOOD flat ~+0.00–0.007) → **skip N∈{6,8}** per the pre-registration; run the **minimum viable set** for seeds {1,2}: pairs at **N∈{2,4}**, pair at **N=30**, **+ GRPO**.
- Before the final read-out: (a) run **GRPO** seed 0 (+1,2), (b) build the **KL harness**, (c) settle the extraction-confound reporting.
