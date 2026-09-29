# Countdown battery — ES finally learns; GRPO still learns more, ~7× faster (2026-07-30)

**Question (user-requested, direction inverted vs all prior arms):** train on countdown, evaluate
math OOD. Does ES learn when the task has headroom (base countdown = 0.08), and what does each
method pay on held-out math?

**Setup.** Qwen2.5-3B-Instruct, 100 steps matched (NOT wall-clock), cap 2048 train+eval,
TRAIN_GPU_MEM_UTIL=0.50 all arms, seed 0. Arms: (1) LoRA-ES r16 all-7-targets vanilla N30 B100
σ=.015 α=.0075; (2) full-param ES N30 B100 σ=1e-3 α=5e-4; (3) verl GRPO TRAIN_BS16×G8
(128 rollouts/step), raw-context prompt parity via identity-chat-template model copy.
Driver `run_countdown_battery.sh` → `results/countdown_battery/`. Session disconnect killed arm 1
at step 39; resumed exactly via new replay-resume in `src/es_lora_main.py` (all 40 replayed
update norms matched the log bit-exact; `--fresh` to opt out).

## Headline

**First regime in this program where ES learns.** Countdown ID accuracy (cap300, paired McNemar
vs base on identical problems):

| arm | countdown ID | Δ vs base | z | math-OOD pooled z (5 sets) | KL proxy | s/step |
|---|---|---|---|---|---|---|
| base | 0.080 | — | — | — | — | — |
| LoRA-ES λ1 | 0.427 | +0.347 | **+9.3** | **−4.56 (real damage)** | 0.078 | 252 |
| LoRA-ES λ0.5 | 0.393 | +0.313 | +9.0 | −2.41 | — | — |
| full-ES | 0.450 | +0.370 | **+9.9** | −1.46 (ns) | 0.017 | 311 |
| GRPO | 0.560 | +0.480 | **+11.6** | −1.55 (ns) | — | ~35 |

Head-to-head on countdown: GRPO > LoRA-ES z=+6.0, GRPO > full-ES z=+5.2, full-ES ≈ LoRA-ES
z=+1.1 (ns).

## Findings

1. **Task, not method, was the blocker.** Same ES code that was inert/destructive on GSM8K/MATH
   learns countdown: LoRA-ES fit 0.057→0.226 (best 0.262 @97, slope +2.0e-3/step), median
   split-half ρ = **0.51** sustained over 100 steps (vs ~0 in every math regime, and vs the
   B=200 math arm where ρ≈0 even at huge batch). Countdown's binary reward has real member-ranking
   signal because base competence is low and improvements are incremental; GSM8K/MATH binary
   reward at near-ceiling did not.
2. **GRPO still dominates**: +0.13 more ID than the best ES arm at matched steps, at ~7× less
   wall-clock/step (35 vs 252/311 s). GRPO train reward 0.13→0.55; responses lengthened
   652→~1200 tok (clip ratio 0.07→~0.4 at cap 2048 — some pressure, but no brevity pathology and
   the gain is real on eval at 2048).
3. **OOD cost is where the arms differ.** LoRA-ES λ1 pays a REAL math forgetting cost
   (pooled z=−4.56, all 5 sets negative; gsm8k −0.057 z=−2.8, math500 −0.043 z=−2.0). Its KL
   grew 0.015→0.078 over training — 4.6× full-ES's 0.017 for LESS ID gain: the r16 adapter
   concentrates the same behavioral shift into a far larger distribution move. λ=0.5 shrinkage
   halves the damage (pooled −2.41) while keeping +0.313 ID — the shrinkage frontier finally has
   a real trade-off to navigate instead of just undoing noise. Full-ES and GRPO preserve math
   ≈ base (pooled ns; only olympiadbench marginal −2.1 for full-ES).
4. **Full-param ES is the surprise quality arm**: same ID gain as LoRA-ES (0.45 vs 0.43, ns
   difference) with 4.6× less KL and no significant math damage — but it is also the slowest
   (311 s/step) and holds the fp32 base copy.
5. Base row reproduced `exp3b_math/base_3b.json` exactly again (protocol determinism holds).

## Caveats

Single seed (3-seed rule not yet satisfied for any cross-arm conclusion). Steps-matched, not
wall-clock-matched: at matched wall-clock GRPO would get ~7× more steps — the gap understates
GRPO's advantage. ES batch geometry (30×100=3000 gens/step) vs GRPO (128 rollouts/step) differ;
per-generation ES is ~1.4-1.9× cheaper but needs 23× more of them. countdown eval slice [:300]
is held out from training rows [300:] but same distribution — "ID" here means same-task held-out.

## Artifacts

`results/countdown_battery/{loraes,fulles,grpo}/` — arm.jsonl (per-step telemetry incl.
correctness bitmaps), arm_summary.json, ood_eval.json, per-question `*_perq.jsonl` for every
set×arm (paired analysis in this report computed from these). LoRA final adapter:
`loraes/theta_final.pt` (durable); full-ES ckpts /tmp only. Logs
`logs/countdown_battery/` (loraes_train.log.part1 = pre-disconnect portion).
