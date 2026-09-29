# Track A — Phase A1 gate: LoRA-ES training sanity (STOP — negative)

**Verdict: LoRA-ES does NOT show learning at the pre-registered settings.** Pipeline is correct
and stable (A0 isolation passed; no collapse after the lr-scale fix; ES signal present), but the
sigma 3-point sweep found no setting where ID rises. Model: Qwen2.5-1.5B-Instruct, GSM8K (base ID
0.855), N=16, 300 steps, 1 seed, alpha=2e-3.

## sigma sweep
| sigma | ID 0->300 | id_gain | KL_final | within-step fit_std | member fit_mean | zero_upd |
|---|---|---|---|---|---|---|
| 0.01 | 0.855->0.820 | -0.035 | 0.0009 | 0.089 | 0.816 | 0.00 |
| 0.02 | 0.855->0.835 | -0.020 | 0.0026 | 0.149 | 0.532 | 0.00 |
| 0.05 | 0.855->0.865 | +0.010 | 0.0000 | 0.001 | 0.000 | 0.97 |

## Diagnosis
- **Perturbation scale too high.** sigma=0.02 degrades members to fit_mean 0.53 (vs base 0.855);
  sigma=0.05 destroys them (0.00, 97% dead/tied steps). ES then optimizes among broken members.
- **Only sigma=0.01 keeps members near base** (fit_mean 0.816, spread 0.089) — but ID still drifts
  DOWN (-0.035): with B=8 prompts/step the z-scored ES gradient overfits each step's batch instead
  of improving held-out accuracy. KL stays ~0 (policy never moves coherently).
- Net: the useful sigma is < 0.01, AND B=8 is likely too small for a low-variance ES gradient.

## Options (need human decision — beyond "tune sigma")
1. Lower-sigma re-sweep {0.002, 0.005, 0.01} (still A1's sigma knob) — keeps members near base.
2. Larger per-step batch B (16/32) to cut ES gradient variance / overfitting (deviation).
3. Larger N and/or more steps (ES in a 9.4M-param adapter space with N=16 is under-populated).
4. Accept LoRA-ES-doesn't-learn-here and reconsider Track A scope.
Mirror LoRA-GRPO lr sweep HELD pending this decision (comparing a non-learning ES to GRPO is moot).

## Round 2 (lower sigma + B=32, human-approved) — STILL NEGATIVE
| sigma (B=32) | ID curve | gain | KL_final | member fit_mean | within-step fit_std |
|---|---|---|---|---|---|
| 0.002 | 0.855->0.825 | -0.030 | 0.0031 | 0.838 | 0.030 |
| 0.005 | 0.855->0.860 | +0.005 | 0.0043 | 0.832 | 0.039 |
Lower sigma fixed member-degradation (fit_mean 0.83 vs 0.53/0.00), but ID still flat and KL~0.
ES update verified correct (inject/commit noise match; z-score sign; eval uses true mean adapter).
**Consolidated A1 finding:** across sigma in {0.002..0.05} and B in {8,32}, LoRA-ES does not improve
held-out GSM8K — the per-step reward differences among viable members don't generalize, so Θ
random-walks near base (KL~0). The A0 speed win is moot without learning; Track A premise in question.
