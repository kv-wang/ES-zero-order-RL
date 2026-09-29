# The eval-length confound: LoRA-ES B=200, GRPO, and a retraction

**Date:** 2026-07-28 · **Model:** Qwen2.5-3B-Instruct · **Program:** `experiments/2026-07-24_es_improvement`

---

## Executive summary

Two headline results in this program turned out to be artifacts of one variable — the eval
generation cap — and neither is a code defect.

1. **The overnight B=200 LoRA-ES MATH arm's "first positive transfer" (ΔOOD +0.0155) is
   withdrawn.** It was measured at `max_tokens=512`. At 2048 the same adapter is −0.0068 OOD and
   its ID primary metric is **0.4470 — identical to base to four decimal places**.
2. **GRPO's apparent degradation is a train/eval mismatch, not a broken pipeline.** GRPO trained
   under a 512-token response cap and is enormously better than base *at 512* (OOD-avg +0.045,
   MATH500 +0.11, L3-5 +0.124). Evaluated at 2048 it falls below base.
3. **The method ranking inverts with the eval cap.** At 512: GRPO > LoRA-ES > base. At 2048:
   base > LoRA-ES > GRPO. Whichever method's training regime matches the eval budget wins.
4. **No trained model beats the untrained base evaluated with enough tokens.** Best trained arm at
   512 is GRPO at OOD-avg 0.3467; base at 2048 is 0.3750.
5. **The cap is a bigger effect than anything training has produced.** Moving base from 2048 to
   512 costs 0.0735 OOD-avg. The largest training effect measured anywhere in this program is
   ±0.045.

What survives from the overnight run: the **shot-noise floor is genuinely gone at B=200**
(split-half Spearman ρ +0.46, zero sub-quantum steps) and **LoRA-ES's wall-clock advantage
reverses at large B** (1.36× slower than full-param). Both are measured on training internals and
timing, not on this eval, so the confound does not touch them.

---

## 1. How the confound arose

Two eval protocols exist in the codebase and they were compared to each other:

| | overnight battery (`overnight_eval.py`) | old battery (`eval_core.eval_on_llm`) |
|---|---|---|
| max_tokens | **512** | **2048** (`eval_base.py:23`, `eval_grpo_ood.py:46`) |
| rows | full if n≤1500, else 500-row subset | cap-300 prefix |
| OOD sets | capability window [0.05, 0.85] | fixed list |

Base accuracy under the two is not comparable — MATH500 0.356 vs 0.530, olympiadbench 0.096 vs
0.190 — because at 512 tokens the base model never reaches an answer on most hard problems.
Base extract-rate at 512 vs 2048: math500 0.520/0.783, minerva 0.404/0.570, olympiad **0.297/0.743**.

The overnight report did flag "different protocol, do not compare digit-for-digit", but the
+0.0155 ΔOOD was computed against a base measured *inside* the 512 protocol, so it looked
internally valid. It was not: at 512, "learned to stop before the cap" and "learned to solve the
problem" are the same measurement.

## 2. The full matrix

All rows: cap-300 prefix, greedy, same reward parser, evaluated this session.
OOD-avg = mean(math500, svamp, minerva, olympiad, countdown) — the definition in
`exp3b_math/OOD_COMPARISON_MATH.csv`.

| arm | OOD-avg | L3-5 (ID) | math500 | svamp | minerva | olympiad | countdown |
|---|---|---|---|---|---|---|---|
| **@512** | | | | | | | |
| base | 0.3015 | 0.2212 | 0.3400 | 0.9100 | 0.1140 | 0.0633 | 0.0800 |
| LoRA-ES step36 | 0.3216 | 0.2995 | 0.4033 | 0.9167 | 0.1213 | 0.0967 | 0.0700 |
| **GRPO** | **0.3467** | **0.3456** | 0.4500 | 0.9200 | 0.1434 | 0.1200 | 0.1000 |
| **@2048** | | | | | | | |
| **base** | **0.3750** | **0.4470** | 0.5300 | 0.9133 | 0.1618 | 0.1900 | 0.0800 |
| LoRA-ES step36 | 0.3682 | 0.4470 | 0.5267 | 0.9167 | 0.1544 | 0.1767 | 0.0667 |
| GRPO | 0.3609 | 0.3917 | 0.4900 | 0.9200 | 0.1581 | 0.1333 | 0.1033 |

SEs: math500 0.029, olympiad 0.023, minerva 0.022, svamp/countdown 0.016, L3-5 ≈0.035.
(amc23 excluded from the table — n=40, SE 0.078, its ±0.125 swings are noise.)

**Validity.** Both base@2048 and GRPO@2048 reproduced their stored results
(`base_3b.json`, `grpo_full_3b.json`) **exactly on all 7 datasets**, in separate processes. The
protocol is deterministic and cross-run comparison is sound.

### Extract rate — the mechanism, in one row

| arm | olympiadbench extract |
|---|---|
| base @512 | 0.297 |
| LoRA-ES @512 | 0.373 |
| GRPO @512 | **0.633** |
| base @2048 | 0.743 |
| LoRA-ES @2048 | 0.727 |
| GRPO @2048 | 0.763 |

At 512 the trained models differ from base mainly in *how often they produce a parseable answer
at all*. At 2048 all three sit at 0.73–0.76 and the differences vanish. Both training runs were
learning to fit the window.

## 3. LoRA-ES at B=200

**Setup.** Phase A→D chain, seed 42, N=30, B=200, σ\*=0.015 / α=0.0075 (flip-rate matched to the
full-param reference), MATH L3-5, deadline-cut at 36/300 steps.

**What stands:**

- **The shot-noise floor is gone.** Median split-half Spearman ρ **+0.459** (MATH) / +0.311
  (GSM8K); 0/36 and 0/57 sub-quantum steps; 15–18 distinct fitness levels among 30 members;
  0 zero-update steps. At B=8 the population spread sat below one quantum on 88–98% of steps and
  ρ was not measurable at all. Phase 0's diagnosis was right, and the estimator now has real
  signal. This is unaffected by the eval confound.
- **The speed claim reverses.** Iso-everything at B=200: LoRA-ES 136.4 s/step vs full-param ES
  100.2 s/step — **1.36× slower**, where A0 measured 2.7–3.6× *faster* at B=8. At small B the GPU
  is starved and batching all N members wins; at B=200 full-param already saturates it and the
  multi-adapter punica overhead remains. LoRA-ES buys ~11 GB VRAM (no fp32 base copy), not time.
- **Raising B is cheap.** Throughput 1,880 → 13,100 tok/s (7×), so a B=200 step costs ~2.4× a B=8
  step, not 25×. Phase 0's "~117 h B-sweep" estimate was a large over-estimate.

**What is withdrawn:** the ΔOOD +0.0155 transfer claim. At 2048 the adapter is −0.0068 OOD, L3-5
exactly equal to base, every per-dataset delta inside 1.6 SE. Its 512-token gains were extraction
(+0.056/+0.070/+0.088 on math500/minerva/olympiad); at 2048 the same deltas are
+0.010/+0.037/−0.017.

So: **36 steps at B=200, with a genuine ranking signal (ρ +0.46) and KL 8.8e-3, produced zero
measurable capability change.** That is a cleaner negative than any previous phase, because for
the first time the estimator was not noise-limited.

## 4. GRPO — diagnosis

The question was whether GRPO scoring below base after 200 steps indicates a bug. It does not.

**Bugs ruled out:**

- Extract rates unchanged at 2048 (math500 0.783 both, olympiad 0.743→0.763) — merge, chat
  template, and parsing are all intact.
- The training prompt in `train.parquet` is character-identical to eval's `build_prompt`
  (`question + "\n\n" + "Please reason step by step, and put your final answer within \boxed{}."`).
- The training reward *is* the eval metric — `grpo/reward_logged.py` wraps the same `math_reward`.
- Movement is mixed, not uniformly negative: svamp +0.007, countdown +0.023, amc23 +0.05.
- GRPO@2048 reproduced its stored eval exactly.

**What actually happened — `data.max_response_length=512` during training:**

| step | entropy | reward | resp_len | clip_ratio |
|---|---|---|---|---|
| 1 | 0.139 | 0.242 | 488.5 | **0.750** |
| 61 | 0.136 | 0.477 | 387.3 | 0.297 |
| 141 | 0.096 | 0.383 | 393.1 | 0.375 |
| 200 | 0.097 | 0.516 | 345.2 | 0.172 |

At step 1, **75% of rollouts hit the cap** and scored 0 regardless of correctness. Over 200 steps
clip_ratio fell 0.75→0.17 and length 488→345 while reward rose 0.366→0.556 (first/last 20). The
reward gain tracks the truncation collapse: the dominant learned behavior was brevity.

**The 512-token eval confirms this directly.** In its own training regime GRPO is far better than
base — OOD-avg 0.3467 vs 0.3015, MATH500 0.45 vs 0.34, L3-5 0.3456 vs 0.2212 (+0.124), olympiad
0.12 vs 0.063 (nearly 2×). GRPO learned exactly what it was optimized for.

At 2048 the cost lands on the hardest problems, monotone in difficulty:

| MATH500 level | base | GRPO | Δ |
|---|---|---|---|
| L1 | 0.778 | 0.815 | +0.037 |
| L2 | 0.732 | 0.714 | −0.018 |
| L3 | 0.594 | 0.562 | −0.032 |
| L4 | 0.432 | 0.395 | −0.037 |
| L5 | 0.333 | 0.236 | **−0.097** |

Secondary factors: entropy halved (0.158 → 0.075), and 200 steps × 16 prompts = 3,200 draws over
a 2,000-row train set (1.6 epochs) at 128 rollouts/step, versus verl's reference scale of 1,024.

## 5. What this changes

- **Every cross-arm number in this program must state its eval cap.** The cap is worth 0.0735
  OOD-avg on the base model — larger than any training effect yet measured (max ±0.045). Rows
  from different protocols are not comparable, and the direction of the bias depends on which
  side was capped.
- **Training-regime and eval-regime must be matched, or the comparison measures the mismatch.**
  Both methods here optimized under a 512-token budget; one was rewarded for it by the eval and
  one was punished by it.
- **The seed-confirmation run is no longer the right next step.** It would confirm a result that
  has been explained away.
- **The GRPO baseline is stronger than the program has been treating it.** Given a matched budget
  it produces a large, unambiguous gain. The program's claim should be "GRPO learns and does not
  transfer OOD", not "GRPO fails".
- **Still true after all of this: nothing beats base OOD.** Ranked at their best: base@2048
  0.3750 > GRPO@512 0.3467 > LoRA-ES@512 0.3216. Training moves capability *within* a length
  budget; no arm has produced OOD capability that survives an uncapped comparison.

## 6. Recommended next steps

1. **Re-run with matched budgets** — train and eval both at 1024 or 2048. At 512, ~50–80% of base
   MATH-family responses truncate, so the reward signal is dominated by a length constraint rather
   than by mathematics. This is the single change most likely to alter every conclusion above.
2. **If continuing LoRA-ES:** the honest open question is no longer "does it transfer" but "does
   36 steps of a ρ=+0.46 signal go anywhere at all". A single longer seed (300 steps ≈ 16 h)
   answers that far more cheaply than 3 seeds of a retracted result.
3. **Report the B=200 estimator result as the real finding.** "The shot-noise floor was the
   binding constraint at B=8, and removing it does not by itself produce learning" is a clean,
   defensible negative — much stronger than the previous phases, which could always be attributed
   to a noise-limited estimator.

---

**Artifacts.** `results/oldproto/{loraes_math_step36.json, cap512_base_and_loraes.json,
cap512_grpo.json, cap2048_grpo_recheck.json, STEP36_VS_GRPO.md}`, runner `src/eval_oldproto.py`,
logs `logs/oldproto/`. Overnight run: `OVERNIGHT_FINDINGS.md` (finding #3 withdrawn),
`results/overnight/{report.md, results.csv}`.
