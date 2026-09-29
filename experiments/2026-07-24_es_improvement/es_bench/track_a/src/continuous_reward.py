"""Phase 4 continuous-reward shaping (V1). Design LOCKED 2026-07-21: z-score both arms.

Both binary and continuous arms use Q1's z-score shaping (es_train.py):
    z_i    = (f_i - mean(f)) / (std(f) + EPS_Z)          # population std, ddof=0
    coeff_i = (alpha / N) * z_i

Arms differ ONLY in the fitness scalar f (the manipulated variable):
  * binary     : f_i = primary_i = #correct/B                 (byte-identical to Q1)
  * continuous : f_i = primary_i + DELTA * secondary_i        (lexicographic scalar)

primary_i in {0, 1/B, ..., 1} (multiples of 1/B); secondary_i = P-bar in [0, 1].
DELTA = TIEBREAK_MARGIN / B with TIEBREAK_MARGIN < 1, so DELTA < 1/B and the
secondary can NEVER invert a primary gap (worst case: a +1/B primary lead vs a
full 1.0 secondary deficit still stays ordered). Therefore order(f) equals the
lexicographic order by (primary, secondary); and when ALL primaries tie, f's
variance is driven entirely by the secondary => std(f) > 0 => nonzero update,
which is the whole point of the continuous tiebreaker (kills zero-update steps).

Pure NumPy only (no torch/vLLM) so this is unit-testable on CPU.
"""
from __future__ import annotations
import numpy as np

EPS_Z = 1e-8            # matches es_train.py z-score epsilon exactly
ZERO_TOL = 1e-12        # matches es_train.py zero-update tolerance
TIEBREAK_MARGIN = 0.9   # DELTA = MARGIN / B  (strictly < 1/B)


def tiebreak_delta(batch_size: int, margin: float = TIEBREAK_MARGIN) -> float:
    """DELTA such that DELTA * (max secondary span=1) < min primary gap (1/B)."""
    return margin / float(batch_size)


def combined_fitness(primary, secondary, batch_size: int,
                     margin: float = TIEBREAK_MARGIN) -> np.ndarray:
    """Continuous-arm fitness scalar = primary + DELTA*secondary (lexicographic)."""
    p = np.asarray(primary, dtype=np.float64)
    s = np.asarray(secondary, dtype=np.float64)
    return p + tiebreak_delta(batch_size, margin) * s


def zscore_coeffs(fitness, alpha: float, n: int, eps: float = EPS_Z):
    """Q1 shaping. Returns (coeffs, z). std is population std (ddof=0)."""
    f = np.asarray(fitness, dtype=np.float64)
    z = (f - f.mean()) / (f.std() + eps)
    coeffs = (alpha / n) * z
    return coeffs, z


def is_zero_update(fitness, tol: float = ZERO_TOL) -> bool:
    """A step is zero-update iff member-fitness std ~ 0 (all members tie)."""
    f = np.asarray(fitness, dtype=np.float64)
    return bool(f.std() < tol)


def n_distinct_primary(primary, tol: float = ZERO_TOL) -> int:
    """Number of distinct primary (#correct/B) levels among members."""
    p = np.asarray(primary, dtype=np.float64)
    # cluster values within tol
    vals = np.sort(p)
    distinct = 1 if len(vals) else 0
    for a, b in zip(vals[:-1], vals[1:]):
        if b - a > tol:
            distinct += 1
    return distinct


def is_pure_tiebreaker(primary, secondary, tol: float = ZERO_TOL) -> bool:
    """True iff all primaries tie AND the secondary breaks the tie (decides ranking).

    This is the continuous-arm analogue of a Q1 zero-update step: binary would
    have produced std=0 here, but the secondary rescues a nonzero update.
    """
    p = np.asarray(primary, dtype=np.float64)
    s = np.asarray(secondary, dtype=np.float64)
    all_primary_tied = p.std() < tol
    secondary_varies = s.std() > tol
    return bool(all_primary_tied and secondary_varies)
