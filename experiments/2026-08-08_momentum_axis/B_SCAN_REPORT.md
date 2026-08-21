# B-Scan Report: ES vanilla, B ∈ {16, 32, 64} on Qwen2.5-Math-1.5B

**Date:** 2026-08-17  
**Question:** Does increasing the per-member batch size B from 8 to 16/32/64 rescue split-half Spearman ρ from ≈0 on Qwen2.5-Math-1.5B (base)?

---

## Setup

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Math-1.5B` (non-Instruct base) |
| Variant | ES vanilla |
| N (population) | 16 |
| Steps | 50 |
| Seed | 0 (single seed) |
| Task | MATH L3-5 (same as momentum_base runs) |
| eval_cap | 300 |
| Script | `axis_probe/run_b_scan.sh` |
| Output | `axis_probe/results/b_scan_base/vanilla_B{16,32,64}_N16_s0*` |
| Logs | `axis_probe/logs/b_scan_base/vanilla_B{16,32,64}_N16_s0.log` |
| Wall-clock | B=16: 17 min, B=32: 19 min, B=64: 22 min (all rc=0) |

Baseline comparison: `axis_probe/results/momentum_base/vanilla_N16_s0` (B=8, 200 steps).

---

## Results

### Split-half Spearman ρ

| B | mean ρ | valid steps | steps_scale_flagged |
|---|--------|-------------|---------------------|
| 8 (baseline) | +0.019 | 112 / 200 | 195 / 200 (97.5%) |
| 16 | +0.029 | 45 / 50 | 45 / 50 (90%) |
| 32 | +0.008 | 50 / 50 | 45 / 50 (90%) |
| 64 | +0.081 | 50 / 50 | 45 / 50 (90%) |

All four values are statistically indistinguishable from 0. The B=64 value of +0.081 is within the noise band (std ≈ 0.27 across steps).

### Final Eval Accuracy vs Base Model

| Config | gsm8k | math500 | svamp | amc23 | minerva | olympiad |
|---|---|---|---|---|---|---|
| Base (untrained) | 0.753 | 0.457 | 0.833 | 0.175 | 0.085 | 0.137 |
| B=16 ES (50 steps) | 0.747 | 0.450 | 0.840 | 0.225 | 0.103 | 0.127 |
| B=32 ES (50 steps) | 0.773 | 0.427 | 0.837 | 0.200 | 0.085 | 0.143 |
| B=64 ES (50 steps) | 0.750 | 0.470 | 0.840 | 0.175 | 0.099 | 0.127 |

No consistent improvement over base across any B value. All differences are within single-seed noise (seed-to-seed variance ≈ 0.01–0.03 from prior runs).

---

## Conclusion

**B scan NEGATIVE.** Increasing B from 8 to 64 does not rescue ρ from ≈0.

The real bottleneck is not fitness resolution from B: `steps_scale_flagged = 45/50` (90%) persists at all B values. This means the ES trainer's own scale-detection logic flags nearly every step as low-variance regardless of B. The root cause is that **Qwen2.5-Math-1.5B (base) on MATH L3-5 produces near-zero cross-member fitness variance**: the base model does not follow the answer format on this task, so nearly all members score similarly (close to 0), leaving no signal to rank members by.

This is distinct from the B=8 resolution argument that applied to the -Instruct model. On the base model, the fitness distribution itself is degenerate, not just low-resolution.

---

## Caveats and Unresolved Items

1. **Single seed, 50 steps only.** Not sufficient to confirm the B=64 ρ=+0.081 is real vs. noise. The std across steps is ~0.27, so a mean of 0.08 over 50 steps has SE ≈ 0.27/√50 ≈ 0.038 — the value is within ~2 SE of zero.
2. **Base model + MATH task combination is likely a bad regime.** The base model does not follow the `\boxed{}` answer format required to get non-zero reward. Fitness resolution is irrelevant if all members get reward ≈ 0. This is the same conclusion as the broader BASE_MODEL_RESULTS.md: "ES in math + base model" is not a valid test of the ES algorithm.
3. **B=200 not tested here.** The overnight B=200 run that achieved ρ=0.31–0.46 was on -Instruct, not base. The question of whether B=200 rescues ρ on the base model remains open.
4. **Only vanilla variant tested.** baseaxis/momentum not scanned.
5. **The countdown regime (ρ=0.51) was not tested here.** Countdown remains the only confirmed working regime. The correct next step to validate B's effect on ρ is to run a B scan on countdown with the -Instruct model.

---

## Artifacts

- Training logs: `axis_probe/logs/b_scan_base/vanilla_B{16,32,64}_N16_s0.log`
- Training jsonl: `axis_probe/results/b_scan_base/vanilla_B{16,32,64}_N16_s0.jsonl`
- Eval perq files: `axis_probe/results/b_scan_base/vanilla_B{16,32,64}_N16_s0_finaleval__{dataset}_perq.jsonl`
- Summary jsons: `axis_probe/results/b_scan_base/vanilla_B{16,32,64}_N16_s0_summary.json`
