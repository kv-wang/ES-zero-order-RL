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
