# Rank-One Momentum Tilt ES Implementation

**Date**: 2026-08-30  
**Status**: Complete, ready for testing

## Overview

Implemented rank-one momentum tilt as the 4th ES variant in `es_train_axis.py`, applying tilted Gaussian perturbations to ALL population members (not a probe mechanism).

## Mathematical Foundation

### Tilted Perturbation
```
ε_i = σ·ξ_i + λσ·m̂·ζ_i
```
where:
- ξ_i ~ N(0, I_d): isotropic Gaussian noise (seed i)
- ζ_i ~ N(0, 1): scalar Gaussian (seed i+1, independent from ξ)
- m̂: unit momentum direction (||m̂|| = 1)
- σ: perturbation scale (same as vanilla ES)
- λ: tilt strength

### Covariance Structure
```
C = σ²(I + λ²·m̂·m̂ᵀ)
```
Creates a prolate spheroid (雪茄形) around the momentum axis.

### Energy Fraction κ (Kappa)
Dimensionless parameter controlling tilt strength:
```
λ = sqrt(κ·d / (1-κ))
```
where d is parameter count.

- κ = 0: vanilla ES (no tilt, λ = 0)
- κ = 0.2: typical value, 20% of energy in momentum direction
- κ → 1: collapses to 1D (λ → ∞)

### Momentum Accumulation
EMA momentum matching SGD:
```
m_t = β·m_{t-1} + Δ_t
```
where Δ_t is the ES update at step t.

### Effective Population Size
```
N_eff = (Σz_i)² / Σz_i²
```
Measures independent sample count after tilting. N_eff < N indicates correlation introduced by tilt.

## Implementation

### Modified Files

#### 1. axis_worker.py

**New RPC method: `es_get_mom_unit_vec()`**
- Returns normalized momentum direction as dict of tensors
- Computes ||m|| in fp64 for numerical stability
- Returns fp32 tensors for memory efficiency
- Returns None if momentum buffer doesn't exist or ||m|| = 0

**New RPC method: `es_set_tilt_member(seed, sigma, lam, m_hat_dict)`**
- Generates tilted perturbation: ε = σ·ξ + λσ·m̂·ζ
- Uses seed for ξ, seed+1 for ζ (maintains statistical independence)
- Sets worker weights to base + ε

**Modified: `es_commit_update_axis(..., tilt_params=None)`**
- Added tilt_params optional argument: dict with {lam, sigma, m_hat_dict}
- Reconstructs tilted perturbations from seeds during gradient reconstruction
- Backward compatible: tilt_params=None falls back to vanilla reconstruction

#### 2. es_train_axis.py

**New command-line arguments**:
```bash
--variant tilt                    # Select tilt variant
--tilt_kappa 0.2                  # Energy fraction κ (0=vanilla, 0.2 typical)
--tilt_mom_beta 0.9               # EMA decay for momentum buffer
--tilt_warmup 1                   # Steps of vanilla ES before tilt activates
```

**Training loop modifications**:
1. **Initialization**: Call `es_mom_init()` to create momentum buffer
2. **Tilt computation** (each step):
   - Compute λ = sqrt(κ·d/(1-κ))
   - Fetch m̂ via `es_get_mom_unit_vec()`
   - Gate tilt by warmup and momentum existence
3. **Member evaluation**: Call `es_set_tilt_member()` for all N members (no probe)
4. **Gradient coefficients**: Divide by σ (es_set_tilt_member already includes it)
5. **Update commit**: Pass tilt_params dict to es_commit_update_axis()
6. **Momentum update**: Momentum buffer updated automatically in es_commit_update_axis()

**New logging fields**:
- `tilt_kappa`: κ value
- `tilt_lambda`: λ value
- `tilt_active`: whether tilt is currently active
- `mom_norm`: ||m|| when tilt is active
- `n_eff`: effective population size

#### 3. run_countdown_tilt.sh

Launch script for countdown task with tilt variant:
```bash
# Default run (kappa=0.2)
./run_countdown_tilt.sh

# Kappa sweep
KAPPA=0.0 ./run_countdown_tilt.sh  # vanilla ES baseline
KAPPA=0.1 ./run_countdown_tilt.sh
KAPPA=0.3 ./run_countdown_tilt.sh

# Quick test
STEPS=10 B=100 ./run_countdown_tilt.sh

# Dry run (plan only)
DRY_RUN=1 ./run_countdown_tilt.sh
```

## Design Decisions

### 1. Independence of Noise Sources
Use seed for ξ_i, seed+1 for ζ_i to maintain statistical independence between isotropic and tilt components. This ensures the covariance structure is exactly C = σ²(I + λ²·m̂·m̂ᵀ).

### 2. Backward Compatibility
κ=0 degrades to vanilla ES (λ=0, no tilt), allowing single codebase for ablations.

### 3. Numerical Stability
- Compute ||m|| in fp64, store normalized direction in fp32
- Cap λ at 100.0 when κ→1 to avoid overflow
- Graceful fallback when m_hat=None (disables tilt)

### 4. Warmup Gating
Tilt activates after warmup steps to ensure momentum has accumulated history. Step 0 momentum is zero, so tilt would be meaningless.

### 5. Gradient Coefficient Scaling
Coefficients divided by σ because `es_set_tilt_member` already multiplies by σ. The gradient estimate is:
```
ĝ = (1/N) Σ_i z_i · ε_i/σ
```
where z_i are normalized fitness scores.

### 6. All-Member Tilt vs Probe
Unlike momentum/baseaxis variants that use 4 probe members + N-4 Gaussians, tilt applies tilted Gaussian to ALL N members. This is conceptually cleaner and allows measuring N_eff directly from the full population.

## Testing Strategy

### Minimal Smoke Test
```bash
cd /home/hyin66/es-fine-tuning-paper/experiments/2026-08-08_momentum_axis/es_bench/axis_probe
STEPS=2 B=8 N=4 KAPPA=0.2 ./run_countdown_tilt.sh
```
Expected: Completes without errors, generates log with tilt_lambda, n_eff fields.

### Ablation: κ=0 Should Match Vanilla
```bash
KAPPA=0.0 SEED=42 ./run_countdown_tilt.sh
# Compare to vanilla run with same seed
```
Expected: Identical fitness curves (tilt_lambda=0, tilt_active=false after warmup).

### Kappa Sweep
```bash
for k in 0.0 0.1 0.2 0.3; do
  KAPPA=$k SEED=0 ./run_countdown_tilt.sh
done
```
Expected: 
- N_eff decreases as κ increases (more correlation)
- Convergence speed may differ
- κ=0 matches vanilla baseline

### Memory Profile
Same as other variants (no additional memory overhead beyond momentum buffer, which is shared with momentum variant).

## Expected Results

### Theoretical Predictions

1. **N_eff < N**: Tilting introduces correlation, reducing effective sample count
2. **Faster convergence** (hypothesis): Concentrating search along momentum direction may accelerate learning in early training
3. **κ sensitivity**: Optimal κ likely problem-dependent; 0.1-0.3 reasonable starting range
4. **Momentum warmup**: Tilt should be ineffective in first few steps (m≈0)

### Logging Diagnostics

Check these fields in training logs:
- `tilt_active`: Should be False for steps < tilt_warmup, True after (if κ>0)
- `mom_norm`: Should grow from 0 and stabilize as training progresses
- `tilt_lambda`: Should be constant = sqrt(κ·d/(1-κ)) when active
- `n_eff`: Should be < N and decrease with larger κ
- `fitness_std`: Should be non-zero (if zero, signal too noisy)

## Files Modified

```
experiments/2026-08-08_momentum_axis/es_bench/axis_probe/
├── src/
│   ├── axis_worker.py          [+50 lines: 2 new RPC methods, 1 modified]
│   └── es_train_axis.py        [+70 lines: args, tilt computation, logging]
└── run_countdown_tilt.sh       [new: 176 lines]
```

## Next Steps

1. **Smoke test**: Run 2-step test to verify no crashes
2. **Ablation**: Confirm κ=0 matches vanilla ES
3. **Kappa sweep**: Measure N_eff and convergence across κ ∈ {0, 0.1, 0.2, 0.3}
4. **Compare to momentum variant**: Tilt vs probe-based momentum acceleration
5. **Update paper**: Document tilt method and experimental results

## References

Implementation follows the rank-one momentum tilt formulation discussed in the research notes. The method is a natural extension of the existing momentum/baseaxis probe architecture, reusing the momentum buffer and seed replay infrastructure.
