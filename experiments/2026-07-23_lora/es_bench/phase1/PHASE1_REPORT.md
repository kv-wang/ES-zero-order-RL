# Phase 1 — Zero-Update Frequency vs Population Size N (Q1)

**Model:** Qwen2.5-Math-1.5B-Instruct · **Task:** MATH levels 3–5 (base acc 0.499,
in [0.2, 0.7]). **Setup:** vanilla ES (no antithetic), z-score fitness shaping,
shared reward object, vLLM greedy generation (max_tokens=512), resident fp32
base + reconstruct θ=base+σε per member. **Config:** B=8 tasks/member,
σ=1e-3, α=5e-4 (repo defaults, held constant across N), data-order seed fixed
across N; **200 steps × 2 seeds** per N. Single GPU per run (GPU0=seed0, GPU1=seed1).

Zero-update criterion: a step is "zero-update" iff member-fitness std = 0 (all N
members tie) ⇒ z-scores all 0 ⇒ measured update L2 = 0. `update_l2` is measured
directly from the fp32 committed parameter delta, not inferred.

## Q1 result — zero-update rate falls sharply with N

| N | zero-update rate (mean ± std over 2 seeds) | per-seed | mean fitness var | s/step (s) | t_eff = s/step ÷ (1−zero) |
|---|---|---|---|---|---|
| 4  | **0.307 ± 0.018** | 0.325, 0.290 | 0.0036 | 5.41 | 7.82 |
| 8  | 0.153 ± 0.012 | 0.165, 0.140 | 0.0045 | 10.81 | 12.76 |
| 16 | 0.085 ± 0.005 | 0.080, 0.090 | 0.0047 | 21.61 | 23.61 |
| 30 | 0.065 ± 0.025 | — | 0.0047 | 40.45 | 43.29 |

**Answer to Q1:** Yes — the zero-update ("ineffective step") rate at N=4 is
**significantly higher** than at N=8/16/30. It is **0.307 at N=4** vs **0.065 at
N=30 (~4.7×)**, and decreases monotonically, roughly halving each time N doubles
(≈ proportional to 1/N — the more members, the less likely all of them tie on
the B=8 batch). At N=4, ~1 step in 3 produces no learning signal; by N=16–30
this drops below 1 in 10.

Practical implication for the effective-throughput view: although raw s/step
grows ~linearly with N (5.4→40.5 s), the *effective* step time `t_eff` (dividing
out wasted steps) is dominated by generation cost, so small N does **not** buy a
proportional reduction in useful-work time — a meaningful fraction of N=4's cheap
steps are wasted. (Caveat: this covers steps 0–200 only; zero-update rates can
shift as training proceeds and fitness variance changes.)

Plots: `plots/zero_update_vs_N.png`, `plots/fitness_var_vs_N.png`,
`plots/sstep_teff_vs_N.png`. Raw per-step data: `es_N{4,8,16,30}_seed{0,1}.jsonl`.

## GRPO control — zero-advantage-group rate
verl GRPO, same Qwen2.5-Math-1.5B-Instruct + **same shared reward object**, 200
steps × 1 seed, group size **G=8**, train_batch=8 prompts ⇒ 64 rollouts/step,
max_response=512. `val_before_train=False` so the reward log is train-only.

| metric | value |
|---|---|
| zero-advantage-**group** rate (per-prompt: all G=8 samples tie) | **0.645** |
| fully-collapsed-**step** rate (all 8 prompts in the step zero-adv) | 0.015 (3/200) |
| mean learning-signal prompts / step | 35.5% |
| GRPO s/step (1.5B, 64 rollouts) | 6.74 (gen 2.39 / logprob-fwd 1.79 / actor-update 1.68) |

### ES vs GRPO — interpreting the two collapse metrics (granularity matters)
The two rates are **not** a like-for-like "wasted step" comparison:
- **ES zero-update** wastes the **entire step** — with only N members evaluated on
  one shared B=8 batch, all members tie relatively often (N=4: 30.7%, N=30: 6.5%).
- **GRPO zero-advantage** is **per-prompt** within a group of G=8 samples of the
  *same* problem. Per-prompt collapse is high (64.5% — many MATH L3–5 problems are
  either always-solved or never-solved by the 8 samples), but because a GRPO step
  aggregates 8 independent prompts, the step is **fully** wasted only 1.5% of the
  time; on average 35% of prompts still provide gradient.

**Takeaway:** at the *whole-step* granularity, GRPO almost never wastes a step
(1.5%) whereas ES at small N=4 wastes ~31% of steps — the small-population ES
regime is where ineffective steps concentrate, and increasing N (or, equivalently,
enriching per-step signal) is the direct remedy. This motivates Q2: is the cheaper
ES step (esp. N=4) still competitive on wall-clock once the wasted fraction and
generation cost are accounted for (see Phase 2).

_Raw GRPO reward log: `es_bench/grpo/reward_phase1_grpo.jsonl`; control timing:
`grpo_control_timing.json`._
