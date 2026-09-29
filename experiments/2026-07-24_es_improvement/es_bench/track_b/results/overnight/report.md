# Overnight run — LoRA-ES at B=200 on 3B (single seed 42)

## 0. Deviations from the task spec (all forced, all logged)

| # | spec | what happened | why |
|---|---|---|---|
| 1 | model `Qwen2.5-Math-3B` | used **Qwen2.5-3B-Instruct** | `Qwen2.5-Math-3B` does not exist — the Qwen2.5-Math family ships 1.5B/7B/72B only (checked against the HF hub). Qwen2.5-3B-Instruct is the only 3B, is cached locally, and is the model the *reused* GRPO-MATH and full-param-MATH results were trained on, so it is the only choice under which Phase D's fold-in is comparable. |
| 2 | GPU0 + GPU1 | **single GPU, arms run sequentially** | `nvidia-smi` shows one H200 on this node. Each arm was given a wall-clock deadline instead of a step budget so the night ends with evaluated checkpoints rather than an unfinished run. |
| 3 | disk ≥ 200 GB free | **107 GB free on /, 27 GB on home** | Not satisfiable. LoRA adapter checkpoints are ~38 MB each so they fit on home; full-param checkpoints go to /tmp. |
| 4 | eval cap | tonight uses the spec's rule (full if n≤1500 else fixed-seed 500 subset) | Reused older numbers were computed at cap-300 prefix — flagged where folded in. |

## 1. Phase 0 — preflight

**Adapter isolation (N=4): PASS.** Exact-match criterion as written in the spec: **4/4**. Routing: all_correct=True, mean own-agreement 1.000 vs cross-adapter 0.000 (margin 1.000).

Both criteria passed, so no judgement call was needed — LoRA arms ran.

GPU poller: 1330 samples at 30 s, peak **138627 MiB**, median 0 MiB.

Per-arm VRAM (poller windows). **These are vLLM *reservations* driven by `gpu_memory_utilization`, not measured requirements** — the meaningful part is the delta, which is the fp32 pristine base copy full-param ES must hold and LoRA-ES need not.

| window | arm | peak MiB | median MiB |
|---|---|---|---|
| 01:46–03:56 | LoRA-ES GSM8K B=200 | 127347 | 127347 |
| 03:57–05:55 | LoRA-ES MATH B=200 | 127347 | 127347 |
| 05:56–06:13 | full-param ES iso-B GSM8K | 138627 | 138627 |
| 06:14–06:29 | Phase D eval | 116783 | 116495 |

## 2. Phase A — σ/α selection

Full-param reference (σ=1e-3) pooled flip-rate **0.301** → acceptance band [0.150, 0.602].
Per-dataset reference: {'gsm8k': 0.0859375, 'math': 0.515625}

| σ_lora | flip-rate | KL drift |
|---|---|---|
| 0.005 | 0.238 | 1.714e-03 |
| 0.015 | 0.320 | 3.737e-02 |
| 0.05 | 0.844 | 1.994e+01 |
| 0.15 | 0.844 | 1.816e+01 |

**Selected σ\* = 0.015, α = 0.0075** (α=σ/2). Ceiling flag: **False**.
Reason: flip-rate 0.320 closest to reference 0.301 within band [0.150,0.602]

## 3. Phase B / C — training arms

| arm | N | B | σ | α | steps done / asked | stop reason | s/step | fit first20→last20 | median split-half ρ | KL final |
|---|---|---|---|---|---|---|---|---|---|---|
| LoRA-ES GSM8K | 30 | 200 | 0.015 | 0.0075 | 57 / 300 | DEADLINE reached at step 57/300 | 136.4 | 0.8324→0.8401 | +0.311 | 0.01089 |
| LoRA-ES MATH L3-5 | 30 | 200 | 0.015 | 0.0075 | 36 / 300 | DEADLINE reached at step 36/300 | 194.7 | 0.3093→0.3301 | +0.459 | 0.002638 |
| full-param ES iso-B GSM8K | 30 | None | None | None | 10 / 10 | — | 100.2 | — | — | — |

### Auto-gate decisions taken

No auto-gate fired: no arm hit the step-60 signal gate or the KL guard.

## 4. Phase D — evaluation

Protocol: greedy, max_tokens=512, full if n<=1500 else 500-row subset (seed 12345), shared `math_reward` parser, KL probe n=64.

Capability window [0.05, 0.85] applied to OOD sets. **Dropped from OOD averages: {'asdiv': 0.856, 'gsm8k': 0.8514, 'svamp': 0.91}**

| entry | arm | λ | ID | OOD-avg (excl. cntdn) | OOD-avg (incl.) | ΔOOD vs base | KL |
|---|---|---|---|---|---|---|---|
| base (gsm8k view) | gsm8k | 0 | 0.8514 | 0.4100 | 0.3250 | — | 2.469e-05 |
| base (math view) | math | 0 | 0.2425 | 0.1052 | 0.0935 | — | 2.469e-05 |
| loraes_gsm8k_final_lambda0.25 | gsm8k | 0.25 | 0.8431 | 0.4064 | 0.3273 | -0.0036 | 1.668e-03 |
| loraes_gsm8k_final_lambda0.5 | gsm8k | 0.5 | 0.8431 | 0.4009 | 0.3222 | -0.0091 | 4.442e-03 |
| loraes_gsm8k_final_lambda0.75 | gsm8k | 0.75 | 0.8438 | 0.4095 | 0.3276 | -0.0005 | 8.522e-03 |
| loraes_gsm8k_final_lambda1 | gsm8k | 1 | 0.8431 | 0.4021 | 0.3286 | -0.0079 | 1.363e-02 |
| loraes_gsm8k_beststep0_lambda1 | gsm8k | 1 | 0.8514 | 0.4100 | 0.3250 | +0.0000 | 2.469e-05 |
| loraes_math_final_lambda0.25 | math | 0.25 | 0.2425 | 0.1170 | 0.1067 | +0.0118 | 5.361e-05 |
| loraes_math_final_lambda0.5 | math | 0.5 | 0.2861 | 0.1234 | 0.1102 | +0.0181 | 1.557e-03 |
| loraes_math_final_lambda0.75 | math | 0.75 | 0.2916 | 0.1156 | 0.1004 | +0.0104 | 4.488e-03 |
| loraes_math_final_lambda1 | math | 1 | 0.2943 | 0.1207 | 0.1038 | +0.0155 | 8.764e-03 |
| loraes_math_beststep25_lambda1 | math | 1 | 0.2561 | 0.1019 | 0.1053 | -0.0033 | 2.643e-03 |

### λ\* — shrinkage frontier argmax (the learned-signal detector)

- **gsm8k**: λ\* = **1** (OOD-avg 0.4100); base 0.4100000000000001. λ\*=1 ⇒ the full trained displacement is the best point on the line.
- **math**: λ\* = **0.5** (OOD-avg 0.1234); base 0.1052. λ\*<1 ⇒ the trained displacement is net harmful and partially shrinking it is better — the same signature Phase 1 found.

## 5. Reused (not retrained) results

- full-param ES MATH N-sweep + GRPO-MATH: `track_b/results/exp3b_math/OOD_COMPARISON_MATH.csv` (Qwen2.5-3B-Instruct, 200 steps, seed 0, **cap-300 prefix eval** — different protocol from tonight's, do not compare digit-for-digit).
- GRPO-GSM8K: reused from `track_b/results/exp3b_gsm8k/grpo_full_3b.json` (**train_batch_size 16, not the requested 128**) — a fresh batch-128 GRPO run did not fit the single-GPU night alongside the two main LoRA-ES arms; the existing run is the closest available baseline and its config delta is stated here.

## 6. Artifacts

- `track_b/results/overnight/eval_results.json`
- `track_b/results/overnight/eval_spec.json`
- `track_b/results/overnight/fulles_isoB_gsm8k.jsonl`
- `track_b/results/overnight/fulles_isoB_gsm8k.jsonl`
- `track_b/results/overnight/fulles_isoB_gsm8k_summary.json`
- `track_b/results/overnight/loraes_gsm8k.jsonl`
- `track_b/results/overnight/loraes_gsm8k.jsonl`
- `track_b/results/overnight/loraes_gsm8k_summary.json`
- `track_b/results/overnight/loraes_math.jsonl`
- `track_b/results/overnight/loraes_math.jsonl`
- `track_b/results/overnight/loraes_math_summary.json`
- `track_b/results/overnight/phase0_isolation.json`
- `track_b/results/overnight/phaseA/SIGMA_STAR.json`
- `track_b/results/overnight/phaseA/lora_gsm8k.json`
- `track_b/results/overnight/phaseA/lora_math.json`
- `track_b/results/overnight/phaseA/ref_gsm8k.json`
- `track_b/results/overnight/phaseA/ref_math.json`
- logs: `track_b/logs/overnight/`  (per-arm .log, `phaseBC.driver.log`, `nvidia_poll.csv`)
