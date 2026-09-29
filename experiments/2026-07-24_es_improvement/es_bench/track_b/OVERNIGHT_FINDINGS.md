# Overnight findings — LoRA-ES at B=200 (3B, seed 42)

> **RETRACTION (2026-07-28, later same day): finding #3 below is withdrawn.** Its eval used
> `max_tokens=512`; re-evaluated at 2048 the step-36 adapter is −0.0068 OOD vs base with L3-5
> identical to base (0.4470). The gain was extract-rate under a binding cap, not capability.
> Findings #1 (shot-noise floor gone at B=200) and #2 (speed advantage reverses) are unaffected.
> See `REPORT_2026-07-28_protocol_and_comparison.md` and `results/oldproto/STEP36_VS_GRPO.md`.

Companion to the mechanical deliverable `results/overnight/report.md` (+ `results.csv`). Three
findings, in order of how much they change the program.

---

## 1. The shot-noise floor is gone at B=200 — Phase 0's diagnosis was right

Phase 0 measured that at B=8 the population fitness spread sat **below one quantum (1/B) on
88–98% of steps**, collapsing 30 members onto 2–4 distinct values. That made the member ranking
shot noise, which is why no (σ, α) could rescue LoRA-ES: there was nothing coherent to point at.

At B=200 (quantum = 0.005):

| arm | median fitness spread | in quanta | steps below 1 quantum | distinct levels (of 30 members) | median split-half ρ | steps with ρ>0.1 |
|---|---|---|---|---|---|---|
| GSM8K | 0.0350 | **7.0** | **0.000** | 15 | **+0.311** | 86% |
| MATH L3-5 | 0.0390 | **7.8** | **0.000** | 18 | **+0.459** | 100% |

Split-half Spearman correlates each member's fitness on two disjoint halves of the 200 problems.
It is the direct test of "is the ranking real or is it noise", and it was **not measurable at all
at B=8**. At B=200 the ranking reproduces across halves on essentially every step. The estimator
now has signal. This is the single result that unblocks the Track A premise.

Zero-update steps: **0/57 and 0/36** (vs 2.5% at B=8).

## 2. The LoRA-ES wall-clock advantage is an artifact of small B — it reverses at B=200

Iso-everything comparison (GSM8K, N=30, B=200, same model, same night, same GPU):

| arm | s/step | tokens/s | notes |
|---|---|---|---|
| **LoRA-ES** | **136.4** | ~13,100 | one batched generate, 6000 seqs, 1 chunk |
| **full-param ES** | **100.2** | — | 30 sequential generates of 200 seqs |

**LoRA-ES is 1.36× SLOWER than full-param ES at B=200**, where A0 measured it **2.7–3.6× faster
at B=8**. The mechanism is clear and was visible in Phase 0: at B=8 full-param runs 8 concurrent
sequences and the H200 is starved, so LoRA's ability to batch all N members at once (240 seqs) is
a large win. At B=200 full-param already has 200 concurrent sequences per member and the GPU is
fed, so that advantage evaporates while the multi-adapter punica kernels keep their overhead.

Corollary that matters for cost planning: raising B is far cheaper than Phase 0's linear model
assumed. Throughput went **1,880 tok/s at B=8 → 13,100 tok/s at B=200** (7×), so B=200 costs
~2.4× a B=8 step, not 25×. Phase 0's "~117 h" estimate for the B-sweep was a large over-estimate.

VRAM (poller): LoRA-ES 127,347 MiB peak vs full-param 138,627 MiB. **Both are vLLM
`gpu_memory_utilization` reservations, not measured requirements**, so only the ~11 GB delta is
meaningful — it is the fp32 pristine base copy full-param ES must hold and LoRA-ES does not.

## 3. First positive transfer in the program — the MATH arm, at 36 steps

Base vs the trained adapter (capability window dropped `asdiv` 0.856, `gsm8k` 0.851, `svamp` 0.910
as out of [0.05, 0.85], so the MATH OOD average is minerva + olympiadbench):

| entry | MATH500 | L3-5 | OOD-avg | ΔOOD | KL |
|---|---|---|---|---|---|
| base | 0.3560 | 0.2425 | 0.1052 | — | 2.5e-05 |
| λ=0.25 | 0.3540 | 0.2425 | 0.1170 | +0.0118 | 5.4e-05 |
| **λ=0.5 (λ\*)** | 0.3880 | 0.2861 | **0.1234** | **+0.0181** | 1.6e-03 |
| λ=0.75 | 0.3980 | 0.2916 | 0.1156 | +0.0104 | 4.5e-03 |
| λ=1.0 (final) | 0.3940 | **0.2943** | 0.1207 | +0.0155 | 8.8e-03 |
| step-25 ckpt | 0.3600 | 0.2561 | 0.1019 | −0.0033 | 2.6e-03 |

Every λ>0 improves OOD; ID MATH500 rises 0.356→0.394 (+0.038) and L3-5 0.2425→0.2943 (+0.052).
Both OOD components move the same way (minerva 0.114→0.121, olympiad 0.0964→0.1202). The step-25
checkpoint is worse than the step-36 final, so the trend is with training, not against it. KL stays
tiny (8.8e-3), so this is not the drift-and-damage signature of every previous ES arm.

**Do not over-read this.** Single seed, 36 steps, and each component is roughly 1.5 SE:
L3-5 +0.052 (SE≈0.032), olympiad +0.024 (SE≈0.016), OOD-avg +0.016 (SE≈0.010). What makes it worth
taking seriously is not any one number but that ID, both OOD sets, and the step-25→36 ordering all
point the same way at near-zero KL. It needs the 3-seed confirmation before it is a result.

Note λ\*=0.5 < 1: mild shrinkage still helps slightly, but unlike every arm in Phase 1 the frontier
here is **above base at every λ**, i.e. the displacement is net useful rather than net damage.

### The GSM8K arm is null, and the reason is measurable

| entry | GSM8K (ID) | OOD-avg | ΔOOD |
|---|---|---|---|
| base | 0.8514 | 0.4100 | — |
| λ=0.75 (λ\*) | 0.8438 | 0.4095 | −0.0005 |
| λ=1.0 | 0.8431 | 0.4021 | −0.0079 |

Flat-to-slightly-negative everywhere. Two reasons, both independent of ES: the base is
**saturated on GSM8K (0.851)**, and the capability window removed **3 of the 5 OOD sets**
(asdiv, gsm8k, svamp all above 0.85), leaving a 3-set average dominated by unrelated math. Phase A
had already flagged this: the GSM8K reference flip-rate was **0.086 vs 0.516 on MATH**, i.e. GSM8K
answers barely move under perturbation, so ES gets little to select on. GSM8K at 3B is the wrong
task for this question, as the earlier 3B GSM8K comparison also concluded.

Sanity check that validates the whole eval path: the GSM8K "best-fit intermediate" resolved to the
step-0 checkpoint (best_fit_step=2, checkpoints at 0/25/50), and its eval reproduced base **exactly
on all 7 datasets** — confirming Θ₀ is a true no-op and the adapter injection path is clean.

---

## What I would do next, in order

1. **Seeds.** The MATH arm at B=200 is the first thing in this program worth confirming. 2 more
   seeds × 36 steps ≈ 4 h on one GPU.
2. **Let it run.** 36 steps is nothing; the deadline cut it, not a gate. The fitness slope is
   +1.5e-3/step and still rising at the end. A full 300-step MATH arm is ~16 h on one GPU.
3. **Drop GSM8K as a training task at 3B.** Saturated base, perturbation-insensitive answers, and
   the capability window guts its OOD panel. MATH L3-5 is where the signal is.
4. **Re-cost the B sweep.** Phase 0's linear model said ~117 h; measured throughput says B=200 is
   ~2.4× a B=8 step. A proper B ∈ {8, 50, 200} × ρ curve is now cheap and would locate the
   minimum B that clears the shot-noise floor.
5. **Retire the LoRA-ES speed claim at large B.** It holds at B=8 and reverses at B=200; the honest
   framing is "LoRA-ES buys VRAM (~11 GB, no fp32 base copy) and buys speed only when B is small
   enough to starve the GPU."
