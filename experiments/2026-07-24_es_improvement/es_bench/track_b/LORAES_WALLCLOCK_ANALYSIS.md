# Making LoRA-ES fit inside GRPO's wall-clock — what the measurements say

**Goal.** Find a LoRA-ES configuration that reaches GRPO-level OOD at less wall-clock than GRPO.
**Method.** All of this is derived from data already on disk — the per-step timing/token logs and
the logged `correct_bitmap_hex` (the full N×B member-by-problem correctness matrix), which lets
smaller batches be simulated exactly offline. No GPU time was spent on this analysis.

---

## 1. The cost law: cost is sub-linear in N·B and linear in length

Per-step time is generation-bound (inject + update are negligible). Two measured throughput
points — 240 sequences → 1,880 tok/s, 6,000 sequences → 13,100 tok/s — fit

```
throughput(seqs) ≈ 69.0 · seqs^0.603        ⇒        t_step  ∝  (N·B)^0.397 · L̄
```

where L̄ is mean generated tokens per sequence. **This is the central fact for the whole
question:** at 6,000 concurrent sequences the H200 is saturated, at a few hundred it is starved,
so halving N·B does *not* halve the time.

| what you cut | factor cut | wall-clock saved |
|---|---|---|
| N·B | 2× | 1.32× |
| N·B | 4× | 1.73× |
| N·B | 8× | 2.29× |
| **L̄** | **2×** | **2.00×** |

Sequence length is the only lever that pays back proportionally. Reducing N or B — the first
instinct — is the *weakest* lever available.

## 2. What cutting B actually costs, measured

From the 36 logged steps at B=200, subsampling columns 64× per step. `fidelity` is the Pearson
correlation between the z-scores at B′ and at the full B=200 — which, because
Δ = (α/N)·Σ zᵢεᵢ with fixed near-orthogonal εᵢ, *is* cos(Δ_B′, Δ_full), i.e. the fraction of the
full-batch update direction retained.

| B′ | split-half ρ | fidelity | distinct levels | rel. cost | s/step | fidelity/√cost |
|---|---|---|---|---|---|---|
| 8 | 0.051 | 0.400 | 4 | 0.279 | 54 | 0.757 |
| 16 | 0.102 | 0.532 | 6 | 0.367 | 72 | 0.878 |
| 25 | 0.138 | 0.618 | 7 | 0.438 | 85 | 0.933 |
| 50 | 0.213 | 0.760 | 10 | 0.577 | 112 | 1.001 |
| **100** | **0.319** | **0.892** | 14 | 0.759 | **148** | **1.024** |
| 200 | 0.424 | 1.000 | 18 | 1.000 | 195 | 1.000 |

Progress over a fixed budget goes as √(steps)·fidelity, so the figure of merit is
fidelity/√cost. It is **flat between B=50 and B=200 and peaks at B≈100** — worth about a 1.3×
speedup for an 11% loss of update direction. Below B=25 fidelity falls off a cliff for very
little extra saving. **B is a real but small lever: ~1.3×, not the 4× you'd hope for.**

## 3. Racing / successive-halving: tested and rejected

Two-stage estimator — score all N on B₁ problems, re-score only the extremes (top-k and bottom-k,
which carry the large |z| that dominates Δ) on the full batch:

| estimator | seqs | rel. cost | fidelity | fidelity/√cost |
|---|---|---|---|---|
| uniform B=100 | 3,000 | 0.759 | 0.892 | **1.024** |
| racing B₁=50, k=8 | 3,900 | 0.843 | 0.802 | 0.873 |
| racing B₁=50, k=4 | 2,700 | 0.728 | 0.750 | 0.879 |
| racing B₁=25, k=8 | 3,550 | 0.812 | 0.683 | 0.758 |

**Worse than plain uniform sampling at every setting.** The middle members keep a noisy score, and
because z is standardised across the whole population, that noise contaminates the survivors'
weights too. Don't build this.

## 4. The free lever: the search dimension d

ES sample complexity scales as d/N, and **d is enormous**: the current adapter is

```
rank 16 × {q,k,v,o,gate,up,down} × 36 layers  =  29.93 M parameters
```

The MLP targets dominate (gate/up/down are 2048↔11008; attention is 2048↔2048 with GQA k/v at
2048↔256). Cutting them costs essentially nothing in generation time — LoRA is ~1% of forward
FLOPs — so this is **free** in wall-clock while directly reducing the number of steps needed:

| config | d | reduction |
|---|---|---|
| current: r16, attn+mlp | 29.93 M | 1.00× |
| **r16, attn-only** | **7.37 M** | **4.06×** |
| r8, attn+mlp | 14.97 M | 2.00× |
| r8, attn-only | 3.69 M | 8.12× |
| r16, q+v only | 3.69 M | 8.12× |
| r4, attn-only | 1.84 M | 16.24× |

At N=30 the current d/N ratio is ~10⁶. Attention-only at rank 8 brings it to 1.2×10⁵. This is the
single largest available improvement to ES's statistical efficiency and it is invisible to the
per-step cost.

## 5. The second free lever: antithetic sampling

`es_lora_main.py:184` draws N independent seeds — each member is an independent Gaussian. There is
no mirroring. Switching to N/2 antithetic pairs (±ε) gives a central-difference estimator at
**identical generation cost**: the linear term is estimated with roughly half the variance and the
fitness-mean bias term cancels exactly. This is a few lines in `lora_es_worker.inject_members`
(negate ε for odd members) plus the matching sign in `commit`. The base-axis probe already
demonstrated the antithetic machinery works in this codebase.

## 6. Putting it together against GRPO's budget

> **MEASURED 2026-07-28 — this section's two input numbers were both wrong, in opposite
> directions, and the net is much worse for ES.** Superseded values below; originals struck
> through in the table. Sources: `results/probe/length_scaling.json` (microbench, §7.1) and the
> completed GRPO@2048 run's own timing log (`logs/grpo_t2048/train.log`, 198 steps).

**Correction 1 — the L̄ factor was pessimistic (helps ES).** Measured on 50 MATH L3-5 prompts,
3B greedy:

| cap | L̄ | trunc rate | acc |
|---|---|---|---|
| 512 | 452.1 | 0.520 | 0.340 |
| 1024 | 594.6 | 0.140 | 0.540 |
| 2048 | 676.7 | 0.060 | 0.580 |

L̄(1024)/L̄(512) = **1.315**, not the 1.6 assumed here — raising the cap costs less than feared
because most of the extra budget goes unused. (Note in passing: base accuracy climbs 0.34 → 0.58
purely from the cap. The cap effect dwarfs every training effect ever measured in this program.)

**Correction 2 — GRPO's budget was overestimated 2.7× (hurts ES, much more).** The ~83.8 s/step
figure was GRPO's *first* step, which is warmup-inflated. Over the full run the mean is
**31.07 s/step**, total **1.72 h**, not 4.6 h. ES gets less than 40% of the budget assumed here.

| config | s/step @1024 | ~~steps in 4.6 h~~ | **steps in 1.72 h** | fidelity |
|---|---|---|---|---|
| current N=30, B=200 | 256 | ~~53~~ | **24** | 1.00 |
| N=30, B=100 | 195 | ~~70~~ | **32** | 0.89 |
| N=30, B=50 | 147 | ~~92~~ | **42** | 0.76 |
| N=16, B=100 | 147 | ~~92~~ | **42** | 0.89 (ranking) / lower (gradient) |

So a matched-budget ES arm is **24–42 steps, not 70–90** — i.e. *fewer* steps than the 36 the
overnight run already did, which produced nothing distinguishable from base. The claim this
section was built to support ("a matched-budget arm gets enough steps to matter") does not
survive its own measurement. Adding the two free levers (attn-only rank 8 → d/N down 8×;
antithetic → variance halved) now has to buy the entire result inside ~30 steps.

### 7.1 Reduced-d config: validated

The attn-only rank-8 template is confirmed at **d = 3,686,400** (144 modules), matching §4's
3.69 M prediction exactly, and the in-memory injection path works end-to-end under it. At the
current σ=0.015 the perturbation norm is σ√d = **28.8**, against **82.1** for the r16 attn+mlp
config in use — the 2.85× shrink §7 warned about, confirmed numerically.

But σ should **not** simply be scaled up by 2.85× to match: at σ=0.015 the reduced-d member is
*already* behaviourally live, not inert — L̄ 684.7 vs base 594.6 (+15%), truncation 0.20 vs 0.14,
accuracy 0.46 vs 0.54 (50 prompts, SE≈0.07, so the accuracy drop alone is not significant but the
length shift is). Norm-matching from 28.8 to 82.1 would likely land in the destruction regime
(§D2 put the knee at σ∈(1e-2, 3e-2) for the *larger* d). Re-run Phase A's flip-rate procedure
over σ ∈ [0.01, 0.05] rather than assuming the norm-matched value.

**The honest structural caveat.** GRPO extracts a full gradient from 128 rollouts per step; ES
extracts N=30 scalars from 6,000. That 47× rollout-efficiency gap is inherent to the method, not
to this implementation, and none of the levers above closes it — they are worth perhaps 3–4× in
combination. ES's case has to rest on either (a) the per-rollout cost being lower (no backward
pass, no optimizer state — true, and worth ~11 GB VRAM), or (b) reaching a different optimum than
the gradient does. The B=200 result is that (b) has not yet been demonstrated even once.

## 7. Recommended next experiment

1. **Microbench first (~15 min):** 10 steps at each of {B=100, B=50} × {L=1024} with attn-only
   rank 8, to measure the real s/step and pin the L̄ factor rather than extrapolating it.
2. **Then one matched-budget arm:** N=30, B=100, attn-only r8, antithetic, L=1024, σ re-selected
   by the Phase-A flip-rate procedure (d changed, so the old σ*=0.015 does not carry over), run to
   GRPO's measured wall-clock.
3. **Gate it on the same eval at the same cap** — the whole point of the 2048 GRPO run in flight.

Anything that changes d **must** re-run Phase A's σ selection: σ is calibrated in adapter space,
and ‖σ·ε‖ ≈ σ√d, so cutting d by 8× shrinks every perturbation by 2.8× at fixed σ.

---

**Code.** `src/analyze_batch_scaling.py`, `src/analyze_racing.py` (both offline, no GPU).
**Data.** `results/overnight/loraes_math.jsonl`, `results/oldproto/batch_scaling_math.json`.
