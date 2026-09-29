# Phase 1 — tail averaging vs pure shrinkage (3B, MATH L3-5, full-param ES)

**Verdict: no extractable signal.** Across 8 matched comparisons (4 arms × 2 averaging windows),
tail averaging does not beat the pure-shrinkage control on the target metric. Pooled OOD-avg gap
= **−0.0014 (t = −0.52, 4/8 positive)**; pooled MATH500 L3-5 gap = **+0.0006 (t = +0.05, 4/8
positive)**. Per the pre-stated reading, this is the second branch: the recovery that averaging
produces is the recovery that *stepping back* produces, and nothing more.

One structured exception, discussed in §5: the ID gap is **monotone in N** — averaging helps at
N=4 and hurts at N=30 — which says something about where the oscillation lives, without changing
the verdict.

Artifacts: `results/phase1_tailavg/fulles_N{4,8,20,30}_summary.json`, driver
`run_phase1_tailavg.sh`, code `src/{tail_average.py, tailavg_worker.py, analyze_tailavg.py}`.

---

## 1. Method, and two deviations from the spec

**No checkpoints existed.** `--ckpt_dir` was never passed in `run_exp3b_math.sh`, so there were no
step-160/170/… weight files to load. They were not re-trained. Instead each trajectory was
**reconstructed exactly**: the ES update is a deterministic function of the per-step member seeds
(one `random.Random(pop_seed)` stream) and the per-step fitness vectors (logged to JSONL), because
`es_worker` draws ε per parameter from a `torch.Generator` seeded with the member seed. So

    theta_t = theta_0 + sum_{s<t} Delta_s,   Delta_s = sum_i (alpha/N) z_i^s * eps(seed_i^s)

is replayable with no token generation. Cost: **27 s (N=4) to 120 s (N=30)** per arm, versus
0.43–3.2 h to retrain.

**It is not CPU-only.** ε is drawn on the *CUDA* generator over vLLM's *fused* parameter layout
(`qkv_proj`, `gate_up_proj`), so bit-exact reproduction requires the GPU and a vLLM process. No
training occurs — only RNG and adds.

**Replay is exact, and validated per step.** The reconstructed ‖Δ_s‖ was compared against the
`update_l2` logged during the original run at every one of 200 steps, in all 4 arms:

| arm | max relative error vs logged ‖Δ_s‖ |
|---|---|
| N=4, N=8, N=20, N=30 | **0.0e+00** (all 800 steps) |

θ_final's displacement also reproduces the independently recorded values (N=4: 187.88 vs 187.87
logged; N=30: 71.73 vs 72.0), and restoring θ_0 gives drift 0.0 and KL 3.3e-6. Evals are refused
if this check fails (tol 1e-4).

**A third arm had to be added.** The spec's control grid was λ ∈ {0.25, 0.5, 0.75}, but tail
averaging turns out to shrink only slightly: its effective λ is **0.911–0.927 in every arm**. It
cancels oscillation *inside* the window but cannot undo the ~160 steps of walk preceding it — the
exact caveat the design anticipated, now measured. The specified grid therefore never reaches the
averaging arm's scale, and the two families would not have overlapped in KL at all. Added
`shrink_matched_*`: pure shrinkage at **exactly** ‖θ_avg − θ_0‖, so the two differ *only in
direction*. That is the comparison reported below; KL-interpolation along the frontier is carried
as a cross-check and agrees throughout.

---

## 2. Per-arm frontiers

Base θ₀: MATH500 L3-5 **0.447**, OOD-avg **0.375** (mean of svamp, minerva, olympiad, math500,
countdown — same definition as `OOD_COMPARISON_MATH.csv`).

### N=30 (KL 29.1e-3, the least-damaged arm)
| variant | drift | KL(e-3) | L3-5 | OOD-avg |
|---|---|---|---|---|
| shrink λ=0.25 | 17.9 | 2.09 | **0.442** | **0.375** |
| shrink λ=0.5 | 35.9 | 7.66 | 0.438 | 0.367 |
| shrink λ=0.75 | 53.8 | 16.64 | 0.410 | 0.359 |
| tailavg avg50dense | 65.6 | 24.30 | 0.378 | 0.363 |
| shrink matched (λ=0.914) | 65.6 | 24.52 | 0.415 | 0.357 |
| tailavg avg5 | 66.5 | 24.92 | 0.387 | 0.355 |
| shrink matched (λ=0.927) | 66.5 | 25.20 | 0.406 | 0.357 |
| θ_final | 71.7 | 29.13 | 0.401 | 0.357 |

### N=20
| variant | drift | KL(e-3) | L3-5 | OOD-avg |
|---|---|---|---|---|
| shrink λ=0.25 | 21.7 | 2.61 | 0.429 | 0.369 |
| shrink λ=0.5 | 43.5 | 9.85 | 0.415 | 0.367 |
| shrink λ=0.75 | 65.2 | 21.65 | 0.415 | 0.374 |
| tailavg avg50dense | 79.3 | 29.95 | 0.369 | 0.347 |
| shrink matched | 79.3 | 31.72 | 0.401 | 0.362 |
| tailavg avg5 | 80.5 | 31.26 | 0.387 | 0.347 |
| shrink matched | 80.5 | 32.64 | 0.406 | 0.358 |
| θ_final | 87.0 | 37.91 | 0.419 | 0.359 |

### N=8
| variant | drift | KL(e-3) | L3-5 | OOD-avg |
|---|---|---|---|---|
| shrink λ=0.25 | 33.7 | 4.50 | 0.410 | 0.363 |
| shrink λ=0.5 | 67.5 | 14.65 | 0.415 | 0.371 |
| shrink λ=0.75 | 101.2 | 30.39 | 0.383 | 0.349 |
| tailavg avg50dense | 123.0 | 42.54 | 0.327 | 0.322 |
| shrink matched | 123.0 | 44.49 | 0.313 | 0.320 |
| tailavg avg5 | 124.8 | 44.44 | 0.318 | 0.323 |
| shrink matched | 124.8 | 45.94 | 0.309 | 0.323 |
| θ_final | 135.0 | 54.21 | 0.332 | 0.321 |

### N=4 (KL 137e-3, the destroyed arm)
| variant | drift | KL(e-3) | L3-5 | OOD-avg |
|---|---|---|---|---|
| shrink λ=0.25 | 47.0 | 5.11 | 0.429 | **0.375** |
| shrink λ=0.5 | 93.9 | 22.93 | 0.406 | 0.351 |
| shrink λ=0.75 | 140.9 | 60.66 | 0.327 | 0.321 |
| tailavg avg50dense | 171.7 | 109.05 | 0.272 | 0.296 |
| shrink matched | 171.7 | 104.40 | 0.240 | 0.298 |
| tailavg avg5 | 174.0 | 110.72 | 0.295 | 0.298 |
| shrink matched | 174.0 | 108.46 | 0.240 | 0.290 |
| θ_final | 187.9 | 137.12 | 0.235 | 0.274 |

---

## 3. The matched comparison

| N | window | ID avg | ID matched-shrink | **gap ID** | OOD avg | OOD matched-shrink | **gap OOD** |
|---|---|---|---|---|---|---|---|
| 4 | avg5 | 0.295 | 0.240 | **+0.055** | 0.298 | 0.290 | +0.009 |
| 4 | avg50dense | 0.272 | 0.240 | +0.032 | 0.296 | 0.298 | −0.002 |
| 8 | avg5 | 0.318 | 0.309 | +0.009 | 0.323 | 0.323 | +0.000 |
| 8 | avg50dense | 0.327 | 0.313 | +0.014 | 0.322 | 0.320 | +0.002 |
| 20 | avg5 | 0.387 | 0.406 | −0.018 | 0.347 | 0.358 | −0.010 |
| 20 | avg50dense | 0.369 | 0.401 | −0.032 | 0.347 | 0.362 | −0.015 |
| 30 | avg5 | 0.387 | 0.406 | −0.018 | 0.355 | 0.357 | −0.001 |
| 30 | avg50dense | 0.378 | 0.415 | −0.037 | 0.363 | 0.357 | +0.006 |

| pooled | mean | sd | t | positive |
|---|---|---|---|---|
| gap ID | +0.0006 | 0.0326 | +0.05 | 4/8 |
| **gap OOD** | **−0.0014** | 0.0077 | **−0.52** | 4/8 |

Noise floor: single seed, cap-300 eval → L3-5 SE ≈ ±0.028, OOD-avg SE ≈ ±0.013.

---

## 4. The finding that decides it: shrinkage dominates averaging outright

Forget the matched comparison for a moment and ask which point on each trajectory is *best*. In
**every arm**, the best shrinkage point beats **every** averaging point on **both** metrics:

| N | best shrink (λ) | ID | OOD | best tailavg | ID | OOD |
|---|---|---|---|---|---|---|
| 4 | 0.25 | 0.429 | **0.375** | avg5 | 0.295 | 0.298 |
| 8 | 0.5 | 0.415 | 0.371 | avg50dense | 0.327 | 0.322 |
| 20 | 0.25 | 0.429 | 0.369 | avg5 | 0.387 | 0.347 |
| 30 | 0.25 | **0.442** | **0.375** | avg5 | 0.387 | 0.355 |

**λ=0.25 restores OOD-avg to 0.375 — exactly base — on both N=30 and N=4**, and lands ID within
0.005 of base on N=30. The N=4 catastrophe (ID 0.235, OOD 0.274, KL 137e-3) is almost entirely
undone by multiplying the displacement by 1/4. A trajectory whose damage scales away linearly is
a trajectory carrying no signal worth keeping: **the best available operation on any of these ES
runs is to throw most of it away, and the limit of that operation is the base model.** No variant
in any arm exceeds base on either metric.

---

## 5. The one real structure: the ID gap is monotone in N

| N | mean gap ID | mean gap OOD |
|---|---|---|
| 4 | +0.044 | +0.004 |
| 8 | +0.012 | +0.001 |
| 20 | −0.025 | −0.013 |
| 30 | −0.028 | +0.003 |

The ID gap falls monotonically across all four arms (regression on the 8 points: slope
−0.0026/member, t = −4.64). The OOD gap shows no such trend (t = −0.59).

Mechanistically this is what the drunkard's-walk picture predicts, and it cuts against averaging
rather than for it. The ES update already averages over N members, so the per-step direction at
N=30 is comparatively clean and there is little lateral sway left for tail averaging to cancel —
averaging then merely lags the trajectory and gives back the (small) progress. At N=4 the per-step
direction is dominated by shot noise, so averaging cancels a real amount of it and recovers ID
relative to a same-length step back. **But even where it helps most (N=4, +0.055 ID), the averaged
point sits at 0.295 versus 0.429 for simply shrinking to λ=0.25** — it wins its matched comparison
and still loses badly to the cheaper option. And on OOD, which is what this program is targeting,
there is no N at which averaging helps.

Caveat on this trend: 4 arms, one seed each, and the two windows per arm are highly correlated
(they share ~150 of 200 steps), so the effective n is 4, not 8. Monotone across all four is
suggestive (exact p ≈ 0.08 for a perfect rank ordering at n=4), not established.

---

## 6. Caveats

- **Single seed per arm.** The program's 3-seed rule means no arm-level number here is final. The
  pooled OOD null rests on 8 comparisons drawn from 4 trajectories.
- **Eval-harness offset.** vLLM runs at `gpu_memory_utilization=0.40` here (vs 0.85 in training) to
  fit the fp32 accumulators, and greedy decoding is mildly batch-sensitive: θ_final scores L3-5
  0.4009 versus the 0.4147 recorded at training time (≈3 problems). **All variants within this
  report were evaluated under identical conditions**, so internal comparisons are unaffected;
  comparisons to the older recorded numbers carry ±0.014.
- **LoRA-ES N=16 is not included.** It is adapter-based through a different trainer, and it is
  inert (KL 2.2e-3, drift ≈ 0) — there is no oscillation to average.
- The KL here is the base-greedy NLL-drift proxy, the same estimator used by every other arm in
  this program.

---

## 7. What this closes

Phase 0 established that the ES update is a fixed-length step in a near-random direction. The open
question was whether a coherent drift was hidden inside it. Phase 1 answers: **on OOD, no — at any
N, at either window length, against a control that isolates direction from displacement.** The
apparent recoveries from averaging are shrinkage, and shrinkage done properly (λ=0.25) is strictly
better than any average while landing exactly on base.

This removes "the signal is there but buried in oscillation" as an explanation for the ES results
on 3B MATH. Combined with the Step-3 gate (calibrated σ, swept α, still no learning), the
remaining live hypothesis is the one Phase 0 measured directly: at B=8 the fitness spread is
sub-quantum on 88–98% of steps, so there is little coherent signal in the per-step ranking *to*
extract. That points at Phase 2 (raise B) as the test that still has something to decide.

**STOP — awaiting direction.**
