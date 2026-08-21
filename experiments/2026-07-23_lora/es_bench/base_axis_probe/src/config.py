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

# ---- evaluation ----
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
