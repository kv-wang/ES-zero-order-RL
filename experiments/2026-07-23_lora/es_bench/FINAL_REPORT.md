# ES Small-Population Effectiveness + Wall-Clock Benchmark — Final Report

Model (Phase 1): Qwen2.5-Math-1.5B-Instruct on MATH levels 3-5. Model (Phase 2/3): Qwen2.5-3B-Instruct. 2x H200. Vanilla ES (no antithetic), z-score shaping, shared reward object, vLLM everywhere, fp32-base reconstruction.

## Phase 0 — base accuracy
| task | model | acc | n |
|---|---|---|---|
| GSM8K | Qwen2.5-1.5B-Instruct | 0.772 | 500 |
| GSM8K | Qwen2.5-3B-Instruct | 0.864 | 500 |
| GSM8K | Qwen2.5-Math-1.5B-Instruct | 0.854 | 500 |
| MATH L3-5 | Qwen2.5-1.5B-Instruct | 0.324 | 367 |
| MATH L3-5 | Qwen2.5-Math-1.5B-Instruct | 0.499 | 367 |

## Phase 1 — zero-update frequency vs N (Q1)
| N | zero-update rate | fitness var | s/step | t_eff |
|---|---|---|---|---|
| 4 | 0.307 ± 0.018 | 0.0036 | 5.41 | 7.82 |
| 8 | 0.153 ± 0.012 | 0.0045 | 10.81 | 12.76 |
| 16 | 0.085 ± 0.005 | 0.0047 | 21.61 | 23.61 |
| 30 | 0.065 ± 0.025 | 0.0047 | 40.45 | 43.29 |

GRPO control zero-advantage-group rate: **0.645** (over 200 steps, 64 samples/step).

## Phase 2 — single-step wall-clock, same GPU0 + 3B (Q2)
| arm | seqs/step | s/step | generation | perturb-swap / logprob-fwd | update | peak VRAM |
|---|---|---|---|---|---|---|
| ES N=4 | 32 | 7.76 | 7.51 | 0.134 (swap) | 0.117 | 87995 MiB |
| ES N=8 | 64 | 15.46 | 15.00 | 0.269 (swap) | 0.188 | 87993 MiB |
| GRPO | 64 | 10.18 | 3.03 | 3.05 (logprob-fwd) | 2.80 (fwd+bwd+optim) | 105063 MiB |

_GRPO's actor update (verl `update_actor`) bundles forward+backward+optimizer; verl does not expose a separate backward vs optimizer split. h/seed extrapolations are omitted as steps-to-plateau was not measured (per spec)._

## Phase 3 — dual-GPU ES scaling
| N (split) | single-GPU s/step | dual-GPU s/step | speedup | scaling eff |
|---|---|---|---|---|
| 4 (2+2) | 7.76 | 3.95 | 1.97x | 98% |
| 8 (4+4) | 15.46 | 7.84 | 1.97x | 99% |
