"""Phase 4 locked configuration — single source of truth for all scripts.

Frozen 2026-07-21 (z-score both arms; house budget 200; fp16 throughout).
Any change here is a deviation from the pre-registration -> STOP + human approval.
"""
# ---- model / data (byte-identical to certified Q1 / phase1) ----
MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DATASET = "math"
LEVELS = [3, 4, 5]
DATA_SEED = 1234            # data order held constant across N
TRAIN_SIZE = 2000
BATCH_SIZE = 8             # B: prompts/member (CRN: all members see same B)
SIGMA = 1e-3
ALPHA = 5e-4              # lr
NUM_STEPS = 200            # house budget (certified phase1); NOT 400
WARMUP = 5
MAX_TOKENS = 512
DTYPE = "float16"          # fp16 throughout Phase 4 (matches trained live weights)

# ---- arms ----
POP_SIZES = [2, 4, 6, 8, 30]
SEEDS = [0, 1, 2]
GRPO_ROLLOUTS = 64         # matches ES N=8 rollout volume

# ---- continuous reward (V1 lexicographic tiebreaker) ----
TIEBREAK_MARGIN = 0.9      # DELTA = MARGIN / B  (< 1/B => never flips a primary gap)
# gold-answer rendering for the secondary key: prime the format opener, score the
# canonical answer content tokens (train set is MATH -> boxed regime).
ANSWER_OPENER = "\\boxed{"

# ---- vLLM engine settings: MATCHED across ES and GRPO (added 2026-07-28) ----
# WHY. Cross-method wall-clock numbers were confounded by the engine, not just the method: GRPO ran
# its vLLM rollout at gpu_memory_utilization=0.5 while full-param ES ran at 0.85 and LoRA-ES at 0.6.
# Utilization sets KV-cache size, KV-cache size sets achievable concurrency, and concurrency is
# exactly what t_step ~ (N*B)^0.397 * L_bar is most sensitive to. Any s/step gap measured across
# unmatched settings is part engine, part method, and the two cannot be separated after the fact.
# This is the same class of error as the 512-vs-2048 eval-cap trap: an uncontrolled knob that
# silently decides the ranking.
#
# THE VALUE IS CAPPED BY GRPO, NOT CHOSEN FREELY. verl keeps the FSDP actor resident
# (param_offload=False, optimizer_offload=False; the 3B run peaked at 57.5GB allocated / 76.6GB
# reserved), so vLLM cannot be given the whole card. free_cache_engine=True releases the KV before
# the update, which is what makes 0.50 work at all. 0.50 is the setting GRPO has actually completed
# 200-step runs at; ES comes down to meet it rather than GRPO going up.
#
# CONSEQUENCE: ES s/step measured at 0.85 (A0, the overnight B=200 run) is NOT valid at this
# setting and must be re-measured before any new wall-clock claim. Lower utilization = less KV =
# less concurrency, and ES at N*B=6000 sequences is the arm most exposed to that.
TRAIN_GPU_MEM_UTIL = 0.50   # ES trainers AND verl rollout. Changing this invalidates all s/step.

# ---- evaluation ----
# Eval utilization is pinned SEPARATELY and must NOT be moved to the training value. 0.60 is what
# every stored row was produced at: base_3b.json reproduced EXACTLY on all 7 sets at this setting,
# which is the only reason the old GRPO row stayed reusable instead of needing a re-run. Greedy
# decoding is batch-sensitive -- the Phase 1 replay at 0.40 shifted MATH L3-5 by 0.014 against the
# recorded number. Changing this silently invalidates every baseline on disk.
EVAL_GPU_MEM_UTIL = 0.60
EVAL_MAX_TOKENS = 2048     # NOT 512. See REPORT_2026-07-28_protocol_and_comparison.md.
EVAL_CAP = 300

CHECKPOINTS_FRAC = [0.0, 0.5, 1.0]   # step 0 (base), 50%, 100%
KL_N_PROMPTS = 200                    # fixed prompt set for forward KL(base||theta)
KL_EVERY = 25                         # steps

# Pinned eval dataset sources (HF id, config, split). Minerva default flagged.
EVAL_DATASETS = {
    "math500":       ("HuggingFaceH4/MATH-500", None, "test"),      # ID; L3-5 = primary
    "gsm8k":         ("openai/gsm8k", "main", "test"),              # OOD easy
    "svamp":         ("ChilleD/SVAMP", None, "test"),               # OOD easy
    "minerva_math":  ("math-ai/minervamath", None, "test"),         # OOD hard (default pin)
    "olympiadbench": ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train"),  # OOD hard
    "amc23":         ("math-ai/amc23", None, "test"),               # optional, high-variance
}
ID_PRIMARY = "math500"     # L3-5 subset is the primary ID metric
OOD_LADDER = ["svamp", "gsm8k", "minerva_math", "olympiadbench"]  # easy -> hard
