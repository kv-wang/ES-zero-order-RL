# Phase 0 — Config audit (read-only)

Scope: the 3B MATH-L3-5 comparison (`track_b/results/exp3b_math/`, 200 steps, seed 0) plus the
A1 LoRA sweeps. All numbers recomputed from the committed JSONL logs, not copied from reports.

**Status correction:** the N20/N30 arms are **not running** — they completed on 2026-07-26 and are
already in `OOD_COMPARISON_MATH_REPORT.md`. GPU is idle. D3 is therefore unblocked and is
answered below.

---

## Table 1 — ES fitness generation

| item | full-param ES (`track_b/src/es_train_fullparam.py`) | LoRA-ES (`track_a/src/es_train_lora.py`) | verdict |
|---|---|---|---|
| decoding mode | `temperature=0.0` (greedy), `seed=42` | `temperature=0.0` (greedy), `seed=seeds[i]` | **already greedy in both** — Phase 2/3 "greedy fitness" is a no-op change |
| per-member seed | `random.Random(pop_seed)` → `pop_seeds[i]`, used as *perturbation* seed | `np.random.RandomState(pop_seed)` → `seeds[i]`, perturbation **and** sampling seed | sampling seed is inert under greedy; perturbation seeds fine |
| B actually used | `C.BATCH_SIZE = 8` | `--batch 8` | **B=8 in every 3B arm** |
| CRN (shared prompts) | `step_batches` precomputed from `DATA_SEED=1234`; all members see identical B prompts | same construction, `data_seed=1234` | **CRN correct in both** |
| max_new_tokens | 512 train / 2048 eval | 512 train / 2048 eval | matched |
| prefix cache | `enable_prefix_caching=False` | `enable_prefix_caching=False` | **already off** — Phase 3 item is a no-op |
| member evaluation | sequential: one θ resident at a time, B=8 concurrent seqs | one batched generate, N×B concurrent seqs | full-param cannot batch across members |

## Table 2 — Update conventions  ← **two defects**

| item | full-param | LoRA | reference (OpenAI-ES) |
|---|---|---|---|
| members | θ + σ·ε | Θ + σ·ε | θ + σ·ε |
| shaping | `z = (f−mean)/(std+1e-8)` | same | same |
| coeffs | `(α/N)·z` | `(α/N)·z` | `(α/(N·σ))·z` |
| **1/σ present?** | **NO** | **NO** | yes |
| σ, α used (3B MATH) | σ=1e-3, α=5e-4 (α=σ/2) | σ=5e-3, α=2e-3 (α=σ/2.5) | — |
| A1 sweep | — | σ∈{0.002,0.005,0.02,0.05}, **α=2e-3 fixed** | α must co-vary with σ |

**Defect A — the α/σ coupling is broken, and the code contradicts its own docstring.**
`es_train_lora.py:7-8` documents `Theta += (alpha/(N*sigma)) * sum z_i eps_i`; `es_train_lora.py:131`
computes `(alpha/N)*z`. The 1/σ is absent in **both** trainers. Under the implemented convention α is
a step size on *unit-noise* directions, fully decoupled from σ. Consequence: **the A1 σ-sweep was
never a step-size sweep.** Holding α=2e-3 while σ ranged over 25× changed only how far members
explored, never how far Θ moved. A1's "σ swept, nothing learned" therefore does **not** rule out a
mis-set step size — the single most likely cause of LoRA-ES inertness was never varied.

**Defect B — the update has constant norm, independent of the fitness signal.**
Because z is z-scored to unit std, ‖z‖=√N always, so ‖Δθ‖ ≈ α·√(d/N) on every non-zero step.
Measured over 200 steps, excluding zero-update steps:

| arm | non-zero steps | mean ‖Δθ‖ | CV | range |
|---|---|---|---|---|
| full ES N=4 | 183/200 | 13.8879 | 1.1e-4 | [13.8842, 13.8916] |
| full ES N=8 | 189/200 | 9.8202 | 1.0e-4 | [9.8176, 9.8226] |
| full ES N=20 | 196/200 | 6.2108 | 1.1e-4 | [6.2089, 6.2134] |
| full ES N=30 | 200/200 | 5.0711 | 1.2e-4 | [5.0695, 5.0730] |
| LoRA-ES N=16 | 195/200 | 2.7356 | 1.3e-4 | [2.7346, 2.7368] |

The step length is constant to 4 significant figures. ES takes a **full-size step even when members
are effectively tied** — a fixed-speed random walk with a weak directional bias. This is the
mechanism behind "KL grows, held-out falls, monotone in N", and it is a property of the estimator as
written, not of the task.

## Table 3 — Fitness quantization floor (B=8 → quantum = 1/B = 0.125)

| arm | median fitness std | in quanta | frac of steps < 1 quantum | median distinct fitness levels (of 9 possible) |
|---|---|---|---|---|
| full ES N=4 | 0.0625 | 0.50 | 0.88 | 2 |
| full ES N=8 | 0.0749 | 0.60 | 0.94 | 3 |
| full ES N=20 | 0.0848 | 0.68 | 0.94 | 3 |
| full ES N=30 | 0.0875 | 0.70 | 0.94 | 4 |
| LoRA-ES N=16 | 0.0749 | 0.60 | 0.98 | 3 |

At B=8 the entire population collapses onto 2–4 of 9 possible fitness values, and 88–98% of steps
have a population spread below a single quantum. The ranking is decided by which member happened to
flip one problem out of eight. **The shot-noise-floor premise behind Phase 2 is confirmed and
measured.**

## Table 4 — GRPO config vs VERL reference

| knob | ours (`grpo_ood/run_grpo_train.sh`) | reference | delta |
|---|---|---|---|
| `data.train_batch_size` | **16** | 1024 | **64× smaller** |
| `rollout.n` (group) | 8 | — | 128 rollouts/step total |
| `actor.optim.lr` | 1e-6 | 1e-6 | matched |
| `actor.kl_loss_coef` (β) | 1e-3 | 1e-3 | matched |
| `ppo_mini_batch_size` | 8 | — | — |
| `max_response_length` | 512 | — | — |
| `max_prompt_length` | 1024 | — | — |
| train data | MATH L3-5 (`prep_data.py --levels 3,4,5`) | — | **matches ES filter** (the report's caveat was unfounded) |

Stale comment: the script header still says `TRAIN_BS=8 x GROUP=8, max_response=256`; the code sets
`TRAIN_BS=16, MAXRESP=512`.

## Table 5 — Parser identity  ← **D1's parser branch is pre-answered**

| path | reward fn | prompt construction |
|---|---|---|
| ES training | `es_bench.shared_reward.math_reward` | `data_math.build_prompt` |
| GRPO training | `grpo/reward_logged.py` → same `math_reward` | `prep_data.py`, same `INSTRUCTION`, verl applies same chat template |
| Eval battery | `eval_core.py` → same `math_reward` | `data_math.build_prompt` |

Train and eval share one reward object and one instruction string **by construction**. The
"≪0.59 ⇒ parser/template mismatch" branch of D1 is therefore largely excluded a priori; D1 reduces
to the memorization test. One residual to verify at runtime: verl's internal chat-template
application vs `apply_chat_template(add_generation_prompt=True)` — same tokenizer, not byte-verified.

## Table 6 — LoRA-ES JSONL telemetry (3B MATH, N=16, σ=5e-3, α=2e-3)

| quantity | value |
|---|---|
| zero-update fraction | 0.025 |
| member-fitness std (median) | 0.0749 (0.60 quanta) |
| update norm | 2.7356, constant (CV 1.3e-4) |
| train fitness, first 20 → last 20 steps | 0.3527 → 0.3750 (**+0.022**) |
| own-val ID | 0.34 → 0.40 |
| KL final | 2.18e-3 |

## Table 7 — D3 diffusion law (answerable now; N20/N30 already landed)

| N | mean ‖Δθ‖ | final drift | KL (e-3) | KL·N | ΔL3-5 |
|---|---|---|---|---|---|
| 4 | 12.707 | 187.87 | 137.1 | 548.4 | −0.212 |
| 8 | 9.280 | 135.00 | 54.2 | 433.6 | −0.115 |
| 20 | 6.087 | 86.95 | 37.9 | 758.0 | −0.018 |
| 30 | 5.071 | 71.72 | 29.1 | 873.0 | −0.032 |

- **Displacement law holds:** observed ‖Δθ‖ tracks the predicted α·√(d/N) to within 9% across a 7.5×
  range of N (ratios 1.000 / 1.033 / 1.071 / 1.093).
- **KL law is falsified as pre-registered:** log-log fit gives **KL = 314·N^(−0.718)**, not N^(−1).
  KL·N is not constant — it ranges 434→873 and rises monotonically with N. The pre-registered
  c ≈ 490±60 is matched only at N=4 (548) and N=8 (434); N=20 (758) and N=30 (873) are 1.5–1.8× above
  the band. Interpretation: displacement falls as 1/√N exactly, but KL is sub-quadratic in
  displacement once displacement is large, so KL decays *more slowly* than 1/N.
- **ΔL3-5 vs KL is monotone but not through the origin:** damage falls with KL (−0.212 @ 137 →
  −0.018 @ 37.9) yet N=30 at the *lowest* KL is slightly worse than N=20 (−0.032 vs −0.018). Within
  single-seed noise (±0.03); no arm reaches ΔL3-5 ≥ 0.

## Finding not in any prior report — ES *does* raise train fitness at large N

Train fitness, mean of first 20 vs last 20 steps:

| arm | Δ train fitness | Δ held-out L3-5 |
|---|---|---|
| full ES N=30 | **+0.083** | −0.032 |
| full ES N=20 | **+0.063** | −0.018 |
| LoRA-ES N=16 | **+0.022** | −0.014 |
| full ES N=8 | −0.008 | −0.115 |
| full ES N=4 | −0.125 | −0.212 |

At N ≥ 20 ES optimizes its training objective and fails to transfer — the **same** train-up /
held-out-down signature already documented for GRPO (reward 0.24→0.59, MATH500 0.447→0.392). The
standing "ES does not learn" framing is too strong and should be narrowed to "ES does not transfer at
N ≥ 20; ES is destroyed at N ≤ 8". This changes what the negative result means.

## Cost basis for Phase 2 (measured, N=30, seed 0)

Generation is 97% of step time in every arm, and per-member generation is 1.87 s regardless of N
(B=8 → only 8 concurrent sequences, i.e. the H200 is badly under-fed).

| N | t_step | 200 steps |
|---|---|---|
| 4 | 7.72 s | 0.43 h |
| 8 | 15.38 s | 0.85 h |
| 20 | 38.42 s | 2.13 h |
| 30 | 57.55 s | 3.20 h |

Phase 2 as specified (N=30, 300 steps, B ∈ {8,64,128}) costs, under a **linear-in-B worst case**:
B=8 → 4.8 h, B=64 → 37 h, B=128 → 75 h; total ≈ 117 h on one GPU. The stated "expect ≤2× at B=128"
gate does **not** hold under that model. The true cost is lower — A0 showed 240 concurrent sequences
generating in 6.77 s — because B=8 leaves the GPU idle, so raising B is partly free. But full-param
ES can only batch *within* a member, so the sub-linearity is bounded. **This must be measured with a
10-step microbench before Phase 2 is launched.**

---

## Deliverable summary

Two mechanical defects (broken α/σ coupling; constant-norm signal-independent update) and one
measured floor (B=8 gives sub-quantum fitness spread on 88–98% of steps) jointly account for the
observed behaviour without needing a task-learnability explanation. The pre-registered KL·N = c law
is falsified (exponent −0.72). Parser identity is clean. GRPO's largest deviation from reference is
batch 16 vs 1024.

**STOP — awaiting direction.**
