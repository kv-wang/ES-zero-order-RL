"""Phase 4 locked configuration — single source of truth for all scripts.

Frozen 2026-07-21 (z-score both arms; house budget 200; fp16 throughout).
Any change here is a deviation from the pre-registration -> STOP + human approval.

DEVIATION 2026-08-14 (human-approved): MODEL switched from the -Instruct version to the
BASE version. Rationale: every result in this try through 2026-08-13 ran on
Qwen2.5-Math-1.5B-Instruct, which is itself a GRPO product (72B RM + iterated rejection
sampling + GRPO, arXiv:2409.12122). BASE_VS_TRAINED_REPORT could therefore not separate
"training from an already-optimized point is net-negative" from "training is net-negative".
The base version is the control that separates them, and it is also what the ES paper used
for math (Qwen2.5-Math-7B, base). See JULY_CONFIG_AUDIT.md for how the -Instruct value got
here in the first place: it was inherited by snapshot-copy from the 07-23 try and never
re-examined, four days and ~10 GPU-hours before the audit caught it.

Override per-run with MODEL=... in the environment; the chosen value is recorded in every
summary json's "model" field, and RESULTS_SUFFIX (below) keeps the two models' results in
separate directories so a re-run can never be silently skipped against the other's output.
"""
import os

# ---- model / data (byte-identical to certified Q1 / phase1) ----
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Math-1.5B")

# Base-version generation does NOT stop on its own under the ChatML prompt template:
# build_prompt() emits <|im_end|> at the turn boundary, but the base model's eos_token is
# <|endoftext|>, so vLLM's default stop condition never fires and every sample runs to
# max_tokens. The -Instruct version has eos = <|im_end|> and so never showed this. Every
# SamplingParams in this try passes STOP; it is harmless for -Instruct (the token is its eos)
# and required for the base version. Caveat: a stop STRING is stripped from the output whereas
# an eos TOKEN is kept in token_ids, so re-running an -Instruct arm may differ from its
# 08-09..08-13 result by that one trailing special token. Those results are frozen on disk and
# are not being re-run, but the "bit-identical across runs" property no longer holds for them.
STOP = ["<|im_end|>", "<|endoftext|>", "\nProblem:"]
# "\nProblem:" terminates the base model's few-shot format (data_math.build_prompt): there the
# turn boundary is the start of the next exemplar, not a special token. It cannot fire early on
# a real solution, which never begins a line with "Problem:".

# Results dir suffix, so base-version and -Instruct results never collide. The run scripts
# skip an arm when its summary json already exists; without this, switching MODEL would make
# every arm "already done" and report the OLD model's numbers as the new model's.
_SLUG = MODEL.rsplit("/", 1)[-1]
RESULTS_SUFFIX = {
    "Qwen2.5-Math-1.5B-Instruct": "",       # historical: leaves 08-09..08-13 results in place
    "Qwen2.5-Math-1.5B":          "_base",
}.get(_SLUG, "_" + _SLUG.replace(".", "").replace("-", "_").lower())  # unknown model -> own dir
DATASET = "math"
LEVELS = [3, 4, 5]
DATA_SEED = 1234            # data order held constant across N
TRAIN_SIZE = 2000
# DEVIATION 2026-08-19 (human-approved): raised 8 -> 1024 to align with es-at-scale math
# (batch_size=1024). Countdown drivers pass --batch explicitly and are unaffected.
BATCH_SIZE = 1024          # B: prompts/member (CRN: all members see same B)
MINI_BATCH_SIZE = 64       # vLLM generate chunk when B exceeds this (peak KV memory)
SIGMA = 1e-3
ALPHA = 5e-4              # lr
NUM_STEPS = 200            # house budget (certified phase1); NOT 400
WARMUP = 5
# DEVIATION 2026-08-18 (human-approved): raised 512 -> 1024. Rationale: 3B-Instruct
# greedy MATH/Olympiad naturally needs ~600/800+ completion tokens; a 512 cap truncates
# mid-reasoning and flips method rankings vs a longer budget. Train and eval must stay
# equal. Countdown drivers pass their own --max_tokens / MAXRESP and are unaffected.
MAX_TOKENS = 1024          # training AND eval completion cap (must stay aligned)
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
