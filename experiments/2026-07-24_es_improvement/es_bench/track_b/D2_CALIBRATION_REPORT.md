# D2 — perturbation-scale calibration + α velocity probe (Steps 1–2)

Model Qwen2.5-1.5B-Instruct, MATH L3-5, greedy, seed 0. No training in Step 1; 20 steps in Step 2.
Deliverables: `results/d2_probe/{fullparam_ref,lora_grid,D2_SIGMA_STAR}.json`,
`results/step2_alpha/*`, `results/plots/d2_sigma_calibration.png`.

---

## Step 1 — σ\* by behavioral-scale match

Two matched observables on a fixed 32-prompt set, 8 perturbation draws each: **answer flip-rate**
(fraction of prompts whose extracted final answer changes vs base) and **per-token KL drift**
(`kl.py` base-greedy NLL proxy — the same estimator both arms and all training runs use).

### Sanity

| check | value | meaning |
|---|---|---|
| full-param base-vs-base drift | −2.2e-06 | proxy is unbiased at zero |
| LoRA Θ₀ (B=0) flip-rate | **0.000** | adapter is an exact no-op → **greedy decoding has no flip floor** |
| LoRA Θ₀ drift | −2.2e-06 | matches the full-param zero exactly |

The Θ₀ check matters: it establishes that every flip measured below is real perturbation signal, not
decoder nondeterminism.

### Reference (full-param)

| σ | flip-rate | KL drift |
|---|---|---|
| 3e-4 | 0.301 ± 0.054 | 9.31e-4 |
| **1e-3 (reference)** | **0.398 ± 0.066** | **8.43e-3** |
| 3e-3 | 0.582 ± 0.060 | 8.54e-2 |

KL scales as σ^2.05 — quadratic in perturbation size, as expected.

### LoRA grid (batched multi-adapter, matching how members are scored in training)

| σ_LoRA | flip-rate | KL drift |
|---|---|---|
| 1e-4 | 0.098 ± 0.036 | −1.48e-5 |
| 3e-4 | 0.109 ± 0.038 | −2.24e-6 |
| 1e-3 | 0.223 ± 0.053 | 3.28e-5 |
| 3e-3 | 0.305 ± 0.030 | 5.65e-4 |
| 1e-2 | 0.449 ± 0.044 | 1.08e-2 |
| 3e-2 | 0.707 ± 0.068 | 1.44 |
| 1e-1 | 0.625 ± 0.000 | 17.9 |

σ=1e-1 is **non-monotone in flip-rate** — the adapter is destroyed and extraction itself collapses,
so "flips" stop being measurable. The destruction knee sits between 1e-2 and 3e-2.

### Result

| solve | σ\* |
|---|---|
| match flip-rate 0.398 | 6.55e-3 |
| match KL drift 8.43e-3 | 9.05e-3 |
| ratio | 1.38× — **agree within 2×, calibration well-posed** |
| **adopted σ\*** | **7.7e-3** (geometric mean) |

### Headline

**σ was never the problem.** The σ already in use was 5.0e-3 — within ~1.5× of σ\*. The A1 grid
{0.002, 0.005, 0.02, 0.05} straddled the correct value: 0.005 near-optimal, and 0.02/0.05 sitting
past the destruction knee this curve now locates, which is exactly why those arms showed member
degradation. Since α was pinned at 2e-3 across that entire sweep, **the only axis that mattered is
the one that was never varied** — direct positive evidence for Phase-0 Defect A, and independent
confirmation that A1's negative is void.

---

## Step 2 — α by KL velocity

σ pinned at 7.7e-3; N=16, B=8, vanilla z-score shaping, greedy fitness; only α varies.
Reference: full-param ES N=30 reached KL 29.1e-3 over 200 steps → **1.455e-4/step**.
Target band = 1–10× that = [1.46e-4, 1.46e-3].

| α | KL@10 | KL@20 | velocity/step | ×N30 | in band | ‖Δ‖ | zero-update | Δfit | fitted *p* (KL ∝ t^p) |
|---|---|---|---|---|---|---|---|---|---|
| 2e-3 | 0.00027 | 0.00052 | 2.61e-5 | 0.18 | no — too small | 2.15 | 0% | −0.057 | 0.93 |
| **1e-2** | 0.00702 | 0.02070 | **1.04e-3** | **7.11** | **YES** | 10.74 | 0% | −0.055 | **1.56** |
| 5e-2 | 0.755 | 7.80 | 3.90e-1 | 2681 | no | 53.7 | 60% | −0.126 | 3.37 |
| 2e-1 | 17.22 | 17.22 | 8.61e-1 | 5918 | no | 214.8 | 90% | −0.060 | −0.00 |

(Δfit over 20 steps is meaningless as a learning signal at this horizon — listed only to show the
large-α arms are not merely slow but broken.)

**Freeze mechanism at α ≥ 5e-2.** The adapter is destroyed → every member scores 0 → fitness std → 0
→ zero-update → ‖Δ‖ = 0. At α=2e-1 the run is frozen for 90% of steps with KL pinned at 17.22. This
is Defect B's failure mode in its terminal form.

### Gate conflict (unresolved by the probe as specified)

α=1e-2 is the only in-band arm, but **KL is superlinear in t** (p=1.56), whereas the Step-2 criterion
compares a 20-step average velocity against N=30's 200-step average. Extrapolating:

| α | fitted *p* | predicted KL(300) | vs Step-3 band [1e-3, 3e-2] |
|---|---|---|---|
| 2e-3 | 0.93 | ≈ 6.4e-3 | inside |
| 1e-2 | 1.56 | ≈ 1.4 | **≈47× over the ceiling** (10× the KL of the *destroyed* full-param N4 arm) |

So the Step-2 velocity gate and the Step-3 KL gate select different α (1e-2 vs ≈2.5e-3). The
extrapolation is only 2 points, but it has one external check: α=2e-3 at σ\* predicts KL(300) ≈ 6.4e-3,
and the measured before-fix 200-step arm came in at 1.5e-3 — same order, scaled about as expected for
a 1.5× larger σ.

**Resolution: bracket it empirically rather than pick by argument.** Step 3 runs three arms
(`run_step3_decisive.sh`) — a matched before-fix control, A at α=1e-2 (Step-2 winner), and B at
α=3e-3 (Step-3 band target). Whichever extrapolation is right, one pass decides it.

---

## Note on the Step-3 gate threshold

The ruling's "+0.022" comes from the **3B** N=16 arm. The decisive run is at **1.5B**, where the
before-fix arms gave +0.0263 (N=20) and +0.0256 (N=30) — and no N=16 arm exists at all. The control
arm above supplies a matched-N, matched-step-count threshold so the gate is measured rather than
borrowed across model scales.

**STOP at the Step-3 gate.**
