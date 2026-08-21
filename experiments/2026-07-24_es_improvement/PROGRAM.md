# ES improvement program — pre-registration (2026-07-24)

Mainline (supersedes the base-axis line, which completes its running confirmation and is archived).
Two tracks, each staged with hard gates: **write a one-page report and STOP at the end of every
phase; never cross a gate alone. 3-seed rule for any conclusion (single seed = trend only).**

## Standing discipline (all phases)
- Unified reward object: every method imports the same per-(prompt,completion) reward. Fitness
  AGGREGATION (mean/min) is an ES objective choice, documented per arm; the reward fn never changes.
- Symmetric stacks & tuning: all generation via vLLM for every arm. Every tuning run logged on both
  sides → `ledgers/tuning_ledger.md`. Timing: cuda-sync + perf_counter, discard 5 warmup, ≥20 timed
  steps, mean±std, decomposed (gen / adapter-write / update; GRPO: +logprob / backward / optimizer).
  One-time startup reported separately. VRAM: torch max_memory_allocated + nvidia-smi peak.
- Raw JSONL per run (per step: fitness vectors, update norms, timing breakdown, tokens). commands.log
  for every command. Never delete logs.
- **Prefix caching OFF everywhere.** LoRA changes K/V projections ⇒ identical prompts under different
  adapters have different KV; token-keyed cache sharing across adapters would be a correctness bug
  (same family as the earlier cross-member leak). Verify KV is not shared across lora_ids.

## Track A — LoRA-ES: remove the batch-sharing constraint
- **A0 build+microbench (no training).** LoRA-ES on in-process vLLM: N members = N LoRA adapters over
  one frozen base; per step sample N adapter perturbations, write adapter tensors into engine LoRA
  slots in-memory (no disk), ONE batched generate w/ per-request LoRARequest + per-request seed,
  z-score ES update on adapter params only. LoRA cfg: rank 16, alpha 32, targets = attn{q,k,v,o}+MLP
  (same set for LoRA-GRPO later). σ_adapter is its OWN config (do NOT reuse full-param σ). B=0 init.
  HARD GATE: adapter-isolation unit test (2 adapters w/ known different forced behaviors in one batch;
  each request must reflect its own adapter). Microbench GPU0 Qwen-3B: s/step + VRAM for N∈{4,8,16,30}
  vs LoRA-GRPO step and full-param GRPO step (same model/prompts/max_new_tokens), full decomposition +
  multi-adapter kernel overhead vs N. STOP → report.
- **A1 training sanity (1.5B, 1 seed, ~300 steps).** ID curve rises, pipeline correct, KL logged.
  σ 3-point sweep (mirror: LoRA-GRPO lr 3-point sweep, symmetric budget, both logged). STOP.
- **A2 main comparison (3B, 3 seeds): LoRA-ES vs LoRA-GRPO.** Identical rank/modules/init, unified
  reward, same train (plain GSM8K), same eval (ID + capability-matched OOD + KL), per-seed plateau,
  full wall-clock. GO: LoRA-ES OOD within CI of LoRA-GRPO at h-to-plateau ≤1.5× AND lower KL AND lower
  VRAM. KILL Track A if s/step ≥2× LoRA-GRPO or learning not competitive. STOP.

## Track B — OOD-targeting variants
- **B1 σ dose–response (cheap; may run on idle GPU alongside Track A).** Full-param vanilla ES (existing
  vLLM backend), 1.5B, σ∈{0.5,1,2,4×default}, 2 seeds each, plain GSM8K, standard battery. Output: σ
  vs {ID, OOD, retention, KL}. Read: monotone retention/OOD gain w/ σ at acceptable ID = GO for 3B ext;
  flat/negative = report as-is. STOP.
- **B2 paraphrase asset (CPU/GPU-light).** GSM8K train: K=4 answer-preserving surface paraphrases via
  local Qwen2.5-7B-Instruct. Constraints: numbers+roles unchanged, entities/phrasing changed. Auto:
  number-multiset identical, gold unchanged, dedup. 50-sample manual review → user. Cache JSONL +
  provenance. Contamination ledger: gsm_symbolic/gsm_plus share this family → separate flagged column;
  svamp/asdiv are primary drift-axis endpoints. STOP (user reviews sample before B3).
- **B3 robust-fitness three-arm (3B, 3 seeds; LoRA-ES param from A if A2 passed else full-param ES).**
  (a) ES-mean-aug: fitness = mean_B mean_K; (b) ES-min-aug: mean_B MIN_K; (c) GRPO-aug on augmented
  pool, same total gen budget/step. Same reward; aggregation documented. Log per-member consistency
  (fraction solved under all K). GO: ES-min-aug beats ES-mean-aug on clean drift OOD (svamp/asdiv) by
  >2SD at matched ID AND beats GRPO-aug. REFUTED if min≈mean. STOP.

## Deliverables
results/<phase>/ JSON+CSV + one-page markdown per phase; benchmark tables reproducible from JSONL;
ledgers/{tuning_ledger,contamination_ledger}.md.

## Status
- Base-axis confirmation running to completion (both GPUs), then archived (do not extend).
- A0 build de-risking: vLLM 0.11 LoRA internals under investigation (in-memory multi-adapter injection
  + per-request LoRARequest batching feasibility). Build starts once the API mechanism is confirmed.
