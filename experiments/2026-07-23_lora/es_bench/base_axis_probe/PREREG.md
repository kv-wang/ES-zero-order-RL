# Base-axis probe — pre-registration (frozen 2026-07-23)

A NEW ES variant that adds a base-anchored directional probe to the population. Kept as a
separate arm; never mixed into vanilla-ES comparison arms. Full-weight first (LoRA = later phase).

## Mechanism
Keep a frozen snapshot of the ORIGINAL base θ₀. At each step the current model is θ_t (the
running ES base). The population of N members is composed **deterministically** (no coin):
- **4 anchor members = 2 antithetic pairs** along the base axis (θ₀−θ_t): members at +a₁,−a₁,+a₂,−a₂
  with aₖ ~ U(0, a_max], a_max=0.1. Antithetic pairing makes the anchor's contribution a clean
  central-difference directional derivative `(a/σ)(f₊−f₋)(θ₀−θ_t)` — pool-mean cancels, and negative
  z-scores cannot blindly extrapolate away from base into unevaluated territory.
- **N−4 Gaussian members** (standard σ·ε exploration).

Fitness = **binary** train reward (#correct/B) only — no continuous tiebreaker. Shared z-score
pool over all N. Update = fitness-weighted sum over the ACTUAL perturbation vectors:
  Δθ = (α/N)[ Σ_gauss z_i ε_i  +  (1/σ) Σ_pairs aₖ(z₊−z₋) (θ₀−θ_t) ].

## Constraints
1. Separate pre-registered arm ("base-axis probe"), never mixed into main comparison arms.
2. Fixed composition (2 antithetic anchor pairs + rest Gaussian) — no fair coin, no one-sided sampling.
3. Scale control: aₖ~U(0,a_max], mirrored −a partner, a_max=0.1. Log ‖a(θ₀−θ_t)‖ vs σ√d each step;
   flag if they diverge by >1 order of magnitude (scale mismatch pollutes the shared z-score pool).
4. Update over actual perturbation vectors. Log signed net displacement along the base axis per step;
   its cumulative trajectory (the "learned anchor coefficient") is the headline diagnostic.
5. Full-weight first (persistent fp32 θ₀ snapshot, ~6.2GB @1.5B). LoRA reparametrization is a follow-up.
6. Early-training degeneracy: while ‖θ₀−θ_t‖≈0 anchor pairs tie and contribute ~0 (harmless). Anchors
   applied to the update only once ‖θ₀−θ_t‖ passes a small threshold; logged either way.
7. Hypothesis + gates (honest): fitness sees TRAIN reward only; pulling toward base usually LOWERS it,
   so the expected useful regime is post-saturation drift-pruning.
   - GO: at matched ID accuracy, policy-KL-to-base significantly below a same-budget vanilla-ES control,
     with OOD not worse.
   - NEGATIVE (report as-is): f₊<f₋ persistently and axis displacement ≈0 or negative — the mechanism
     declines to regularize. Do NOT tune rewards to rescue.
8. Per-step logs: (f₊−f₋) per pair, signed axis displacement, ‖θ_t−θ₀‖; at eval points: train acc,
   OOD acc, KL — side-by-side with the vanilla control.

## Pilot config
Qwen2.5-Math-1.5B, MATH L3-5, binary reward, B=8, σ=1e-3, α=5e-4, 200 steps. N=16 (4 anchor + 12
Gaussian). Arms: base-axis probe vs vanilla-ES control (all Gaussian), seeds {0,1}. Eval: OOD/ID
battery (cap 300) + KL-to-base proxy, side by side.

## Amendment (2026-07-23, human-approved after smoke)
Smoke at a_max=0.1 showed the anchor perturbation (‖a(θ₀−θ_t)‖ ≈ 0.3–1.5) was 27–123× smaller
than the Gaussian scale σ√d ≈ 39.3 → f₊−f₋ = 0 on every step (no reward signal), scale flag on 100%.
Fix (approved): **a_max = 1.0** (a~U(0,1], probing up to the full base point; stays 'toward base',
anchor norm scales with drift and matches σ√d in the post-drift regime). All else unchanged.
