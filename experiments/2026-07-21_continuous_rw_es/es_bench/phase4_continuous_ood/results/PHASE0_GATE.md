# Phase 4 — Phase 0 gate report (audit, no training)

Env (certified ES env): **verl** (datasets 5.0.0, transformers 4.57.6, vLLM 0.11).
Design locks: z-score both arms; budget 200; fp16; KL = forward KL(base‖θ).

## 0a — step-0 base accuracy (Qwen2.5-Math-1.5B-Instruct, greedy, house extraction)
cap = 500 rows/dataset (math500 full=500; gsm8k capped from 1319; olympiad capped from 674 after image-filter).

| dataset (role) | n | acc | extract-rate | headroom | ceiling risk (<5pt) |
|---|---|---|---|---|---|
| math500 overall (ID) | 500 | 0.580 | 0.800 | 0.420 | no |
| **math500 L3-5 (primary ID)** | 247 | **0.507** | — | 0.493 | no |
| svamp (OOD easy) | 300 | 0.910 | 0.987 | **0.090** | no (near-ceiling) |
| gsm8k (OOD easy) | 500 | 0.852 | 0.990 | 0.148 | no |
| minerva_math (OOD hard) | 272 | 0.132 | **0.555** | 0.868 | no |
| olympiadbench (OOD hard) | 500 | 0.262 | **0.722** | 0.738 | no |
| amc23 (optional) | 40 | 0.500 | 0.950 | 0.500 | no |

**Flags:**
- No dataset trips the <5pt ceiling rule. **SVAMP (0.91)** has only 9pt headroom → low sensitivity to improvement; keep in ladder but treat as low-power.
- **Extraction reliability is poor on the OOD-hard end**: minerva 0.555, olympiad 0.722. ~28-45% of outputs yield no house-extractable answer, so those accuracies conflate reasoning failure with extractor/format mismatch — a measurement confound on exactly the sets meant to carry the OOD-hard signal.

## 0b — certified-run reuse audit
Certified phase1 (Q1) config: Qwen2.5-Math-1.5B · MATH L3-5 · data_seed 1234 · B=8 · σ=1e-3 · α=5e-4 · **200 steps** · warmup 5 · max_tokens 512 · z-score shaping · seeds {0,1}, N∈{4,8,16,30}.
- **House budget = 200** (use, not 400).
- **Reusable as-is:** ES-binary N∈{4,8,30} seeds {0,1} (6 runs).
- **Must run:** ES-binary N∈{4,8,30} **seed 2**; ES-binary N∈{2,6} all seeds; all ES-cont arms; GRPO seeds {1,2} (Q1 GRPO was 1 seed).

## 0c — unit tests
- **CPU (10/10 PASS):** lexicographic ordering (primary dominates, secondary breaks exact ties, full order), exact-tie→equal coeffs, continuous→binary when secondary constant, pure-tiebreaker→nonzero update, and **Q1 zero-update replay reproducing all 1600 phase-1 flags with zero mismatch**.
- **GPU (ALL PASS):** scoring under member weights does not corrupt live weights or fp32 base; es_restore_base bit-exact; scoring deterministic under fixed weights; scoring reflects live weights (base 0.766 ≠ member 0.769).

## Risks surfaced for later phases
- **Secondary-key saturation:** several per-prompt P̄ = exactly 1.0 on short numeric answers → the tiebreaker may not separate tied-primary members when the base already puts ~all mass on the gold token. Could limit continuous-reward's benefit at small N. To watch in Phase 1/2 via pure-tiebreaker fraction vs actual update norm.
- OOD-hard extraction confound (above) may add noise to the pre-registered OOD-ladder average.

---
## Phase 1 smoke (ES-cont N=4 seed0, 30 steps) — PASS
zero_update_rate=0.0 · pure_tiebreaker_rate=0.333 (≈ Q1 N=4 0.31) · scoring_overhead=5.3% · s/step=5.71 (Q1 5.41) · drift bounded. Mechanism validated: the secondary activates on exactly the steps binary wastes.
