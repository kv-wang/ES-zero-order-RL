# Phase 0 — Read-only Audit + Smoke Tests + Model Selection

Branch: `codex/implement-controlled-ood-evaluation-pipeline` (commit 7a72593).
Hardware: 2× H200 NVL (~143 GB). Env: `/home/hyin66/micromamba/envs/verl`
(verl 0.8.0, vllm 0.11.0, torch 2.8+cu128, transformers **4.57.6** — pinned <5
because vllm 0.11 breaks on transformers 5.x: `Qwen2Tokenizer has no attribute
all_special_tokens_extended`).

## 1. Code audit

### Training entry points / config systems
- **Real ES (vLLM):** `es_fine-tuning_countdown_accl.py` — Ray, one vLLM engine per
  GPU, NCCL weight broadcast. Config = argparse + module constants
  (SIGMA=1e-3, ALPHA=5e-4, POPULATION_SIZE=30). This is the authoritative ES loop.
- **Real ES (HF accelerate):** `countdown/es_fine-tuning_countdown.py`,
  `..._iid.py`, and the two `es_fine-tuning_conciseness*.py` — HF `generate`.
- **OOD pipeline:** `ood_eval/run_experiment.py` — **scaffolding only**: `fake_es_update`
  (adds random noise, no fitness/population) and HF `model.generate`. Not a real
  ES or GRPO trainer.
- **GRPO/PPO:** none in repo. Added separately via **verl 0.8.0** (this work).

### Reward functions
- `countdown/countdown_task.py`: `reward_function` = 0.1·format + answer
  (countdown-specific). Pure string→scalar, reusable.
- For the math task we use a **single shared reward object**
  `es_bench/shared_reward.py::math_reward` (correctness via
  `ood_eval/answer_extraction.extract_final_answer`), imported by BOTH the ES
  trainer and the verl GRPO control (`verl_compute_score` adapter) — satisfies
  protocol constraint 2.

### Fitness shaping — z-score, not rank
All ES scripts use `z = (r - mean) / (std + 1e-8)` (accl line 269, conciseness
line 285). No rank transform, no antithetic/mirrored sampling. Our
`es_bench/es_train.py` keeps the identical z-score shaping.

### Generation backends
- Real ES (accl): **vLLM** (temp=0 greedy, prefix caching **disabled**).
- OOD scaffolding: HF `generate`.
- GRPO (verl): **vLLM** rollout. → All benchmark arms use vLLM (Phase-2 fairness).

### Multi-GPU status
- ES accl: multi-engine via Ray placement groups, 1 GPU/engine, NCCL broadcast
  from engine 0. Our `es_train.py` is single-engine per process, GPU pinned via
  `CUDA_VISIBLE_DEVICES` (Phase 1/2). Phase 3 splits the population across two
  such processes (2+2 / 4+4) with a coordinator over scalar fitness.
- GRPO (verl): native FSDP/Ray multi-GPU (audited in Phase 3, not modified).

### Local model paths (HF cache, persistent JuiceFS)
`Qwen2.5-Math-1.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`,
`Qwen2.5-Math-7B-Instruct`. **`Qwen2.5-Math-3B` does not exist upstream** (Math
family is 1.5B/7B/72B only) — 3B has the Instruct variant only.

## 2. Zero-update criteria
Update = `(α/N) · Σ_i z_i · ε_i`, `z_i = (f_i − mean)/(std + 1e-8)`.
- If **all members share the same fitness** → std = 0 and every numerator
  `f_i − mean = 0`, so `z_i = 0/(1e-8) = 0` exactly ⇒ **update = 0**.
- The `+1e-8` is the div-by-zero guard; because the numerator is exactly 0 when
  std=0, it yields an exact zero update (not a large spurious one).
- Detection in `es_train.py`: `zero_update = (fitness_std < 1e-12) or (update_L2
  < 1e-12)`, and `update_L2` is **measured directly** (fp32 Σ of the committed
  parameter delta in `es_worker.es_commit_update`), not inferred.
- GRPO analogue = **zero-advantage group** (a prompt group whose G samples all
  get equal reward ⇒ group-normalized advantage 0). Logged via
  `es_bench/grpo/reward_logged.py`, reconstructed offline in `analyze_phase1.py`.

## 3. Logging hooks (logging only, no algorithm change)
- ES: per-step JSONL row in `es_train.py` (fitness vector, mean/std/var,
  update_L2, zero_update, tokens, timing breakdown gen/perturb-swap/update).
  Corresponds to accl lines 261–309 (post-normalization, pre/post update).
- GRPO: reward wrapper logs `{gidx, uid, level, split, reward}` per sample.

## 4. vLLM smoke test
`Qwen2.5-Math-1.5B-Instruct`, 8 GSM8K Q, greedy: acc 0.625, extraction 1.0,
1839 tok/s. vLLM load+generate OK in the verl env.

## 5. Base accuracy → model selection
GSM8K (500 Q, greedy pass@1):

| Model | GSM8K | MATH L3–5 (367 Q) | in [0.2,0.7]? |
|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 0.854 | **0.499** (L3 .66/L4 .52/L5 .35) | MATH ✓ |
| Qwen2.5-1.5B-Instruct | 0.772 | 0.324 (L3 .48/L4 .38/L5 .15) | MATH ✓ |
| Qwen2.5-3B-Instruct | 0.864 | — | (Phase 2 timing model) |
| Qwen2.5-Math-3B | N/A (does not exist) | — | — |

**Decision gate:** both 1.5B models exceed 0.7 on GSM8K → GSM8K too easy. Per the
spec's documented fallback, **train on MATH levels 3–5**. Selected Phase-1 model =
**Qwen2.5-Math-1.5B-Instruct** (base 0.499, centered in [0.2,0.7] → maximal
fitness-variance signal for the zero-update study). Confirmed live: at
max_tokens=512 the ES per-step mean fitness sits ~0.41 with genuine zero-update
steps present.

## 6. Phase 1 minimal-change diff plan
New, self-contained `es_bench/` package (no edits to existing repo algorithm):
- `es_worker.py` — vLLM `worker_extension_cls`: resident **fp32 base** +
  reconstruct `θ = base + σε` per member (protocol 6b, no fused bf16 drift);
  prefix caching disabled (6a); direct fp32 update-L2.
- `es_train.py` — vanilla ES loop, z-score shaping, timing (cuda-sync +
  perf_counter), per-step JSONL, VRAM (torch max_allocated + nvidia-smi peak).
- `shared_reward.py` / `data_math.py` — shared reward + MATH L3–5 data (same
  data_seed across N).
- `grpo/` — verl GRPO control (same model + shared reward), reward logging.
- `run_phase1_es.sh` (N∈{4,8,16,30}×2 seeds), `run_phase2.sh`, `analyze_phase1.py`.

**Deviation from spec (documented):** `max_tokens=512` (not 1024) for the ES
sweep — Phase 1 targets frequency/timing (not quality), 512 keeps 8 runs × 200
steps tractable overnight and is applied identically across all N, so the Q1
N-comparison is unaffected. It also pulls fitness off the ceiling seen at 1024.
