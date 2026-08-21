# Tuning-effort ledger (symmetric tuning discipline)
Every hyperparameter tuning run for ANY arm is logged here, on both sides, so no method gets
more tuning budget than its comparator.

| date | track/phase | arm | param swept | values | seeds | outcome/chosen |
|---|---|---|---|---|---|---|
| (none yet) | | | | | | |
| 2026-07-24 | A/A1 | LoRA-ES | sigma (exploration) | 0.01, 0.02, 0.05 | 1 (seed0) | NO LEARNING: 0.02/0.05 degrade members (fit_mean 0.53/0.00); 0.01 near-base but ID drifts down (B=8 overfit) |
| 2026-07-24 | A/A1 r2 | LoRA-ES | sigma + B | sig{0.002,0.005}, B=32 | 1 (seed0) | STILL NO LEARNING: members near-base (fit_mean 0.83) but ID flat (+0.005/-0.030), KL~0.004 (Θ never accumulates); per-step gradient doesn't generalize |
| 2026-07-24 | B1 | full-param ES | sigma (dose-response) | 5e-4,1e-3,2e-3,4e-3 | 1 (seed0) | running (OOD+KL) |
