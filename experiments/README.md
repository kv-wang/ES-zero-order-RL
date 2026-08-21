# Experiments

Each experiment **try** is a self-contained, dated folder under `experiments/`. A try
snapshots its own copy of the `es_bench/` harness package plus all of its results, so
runs stay reproducible and comparable even as the harness code evolves between tries.

## Convention

```
experiments/
  <YYYY-MM-DD>_<short-slug>/
    es_bench/                 # snapshot of the harness package for this try (importable)
      es_train.py, es_worker.py, shared_reward.py, data_math.py, ...
      run_phase*.sh           # drivers; cd/REPO anchored to THIS try folder
      grpo/                   # GRPO (verl) control code + data + outputs
      phase0..3/              # per-phase results, logs, plots, PHASE*_REPORT.md
      FINAL_REPORT.md         # consolidated results for the try
      commands.log            # every command run (protocol requirement)
      report_data.json
    verl_hydra_outputs/       # verl/hydra run dumps (provenance)
```

**Starting a new try:** copy an existing try folder to a new `<date>_<slug>/`, then update
the `cd` / `REPO` anchor lines at the top of each `es_bench/**/*.sh` to point at the new
folder. Because scripts run with the try folder as cwd, `from es_bench...` imports and the
relative `es_bench/...` paths resolve inside the try automatically. Log the new try below.

## Tries

| Date | Folder | Summary |
|---|---|---|
| 2026-07-18 | [`2026-07-18_es-smallpop-vs-grpo-wallclock`](2026-07-18_es-smallpop-vs-grpo-wallclock/) | ES small-population effectiveness (zero-update rate vs N) + single-step wall-clock vs GRPO + dual-GPU ES scaling. Qwen2.5-Math-1.5B / Qwen2.5-3B, 2× H200. See its [`FINAL_REPORT.md`](2026-07-18_es-smallpop-vs-grpo-wallclock/es_bench/FINAL_REPORT.md). |
| 2026-07-21 | [`2026-07-21_continuous_rw_es`](2026-07-21_continuous_rw_es/) | Does a continuous (lexicographic-tiebreaker) reward improve ES OOD at small N? **Refuted** — zero-update goes to 0 as designed but buys no OOD; at N=2 it hurts both OOD and ID with 2.6× KL. Read as evidence *for* noise-as-regularizer. [`REPORT.md`](2026-07-21_continuous_rw_es/es_bench/phase4_continuous_ood/REPORT.md). |
| 2026-07-23 | [`2026-07-23_lora`](2026-07-23_lora/) | **Slug is misleading — contains no LoRA work.** Base-axis probe: 4 of N members become antithetic pairs along θ₀−θ_t. Pilot weakly positive (KL −16%, cum_disp positive, OOD flat); confirmation run **aborted** mid-flight and the pre-registered N=12 control never ran. 4-seed reanalysis added 2026-08-08: [`REANALYSIS_4SEED.md`](2026-07-23_lora/es_bench/base_axis_probe/REANALYSIS_4SEED.md). |
| 2026-07-24 | [`2026-07-24_es_improvement`](2026-07-24_es_improvement/) | Mainline improvement program, 2 gated tracks. Track A (LoRA-ES) **negative**: never improves held-out GSM8K. Track B: matched cross-method comparison, ES update direction ~94% noise on math (split-half ρ=0.06); countdown battery is the first regime where ES learns (ρ=0.51) but GRPO still leads at ~7× less wall-clock. [`PROGRAM.md`](2026-07-24_es_improvement/PROGRAM.md), [`ledgers/experiment_ledger.md`](2026-07-24_es_improvement/ledgers/experiment_ledger.md). |
| 2026-08-08 | [`2026-08-08_momentum_axis`](2026-08-08_momentum_axis/) | **A1 — momentum axis.** Same antithetic central-difference estimator as the base-axis probe, pointed at the recent trajectory (EMA `m_t`) instead of the negated full history. Three parallel arms selected by `--variant {momentum,baseaxis,vanilla}`. CPU regression proves the baseaxis path is bit-identical to the frozen 07-23 worker, so arms merge with that try's seeds. [`PREREG.md`](2026-08-08_momentum_axis/PREREG.md). |

> **Cross-cutting:** [`EXTRACTOR_BUG_REPORT.md`](EXTRACTOR_BUG_REPORT.md) — the `\boxed{}` regex
> cannot match nested braces, which makes 20–34% of math eval questions structurally unscoreable
> and silently drops 23.6% of the MATH L3-5 training pool. Affects **all** tries and both ES and
> GRPO; countdown is unaffected. Paired tests are immune, absolute accuracies are not.
