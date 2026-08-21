import re
from dataclasses import dataclass
from typing import List, Optional

# NOTE on \boxed parsing (fixed 2026-08-09).
#
# This module used to locate boxed answers with r"\\boxed\{([^{}]+)\}". A regular
# expression cannot express balanced delimiters, and that character class explicitly
# forbids braces, so EVERY nested answer silently failed to match:
#
#     \boxed{18}            -> matched
#     \boxed{\frac{1}{2}}   -> no match
#     \boxed{\sqrt{3}}      -> no match
#     \boxed{\text{even}}   -> no match
#
# The failure was invisible rather than loud: no extraction path below can ever
# produce a string containing "{" (FINAL_ANS_RE's character class excludes braces,
# NUMERIC_TAIL_RE is digits only), so any ground truth containing a brace scored
# exactly 0 regardless of what the model wrote. Measured before the fix: 20.7% of
# MATH-500, 25.0% of Minerva and 34.0% of OlympiadBench were unscoreable this way.
#
# Brace matching is now done by counting. The logic is ported from the reference
# implementation in VsonicV/es-at-scale, es_at_scale/reward_function/math_grader.py
# (last_boxed_only_string), with two deliberate deviations documented at _last_boxed.
#
# See experiments/EXTRACTOR_BUG_REPORT.md for the full impact analysis.

BOXED_START_RE = re.compile(r"\\(?:boxed|fbox)")
FINAL_ANS_RE = re.compile(r"(?:final\s+answer\s*[:=]|answer\s*[:=])\s*([-+*/().,\w\\]+)", re.IGNORECASE)
NUMERIC_TAIL_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class ExtractionResult:
    extracted: Optional[str]
    used_boxed: bool
    success: bool


def _boxed_content_at(text: str, idx: int) -> Optional[str]:
    """Content of the braced group opened after position `idx`, or None.

    Returns None when the command has no braced argument (``\\boxed 5``) or when the
    group is never closed (a generation truncated at max_tokens).
    """
    open_idx = text.find("{", idx)
    if open_idx < 0:
        return None
    depth = 0
    for j in range(open_idx, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j]
    return None


def _last_boxed(text: str) -> Optional[str]:
    """Content of the last parseable ``\\boxed{...}`` / ``\\fbox{...}`` in `text`.

    Deviations from the upstream reference, both intentional:

    1. Upstream inspects only the final ``\\boxed`` and gives up if it is unbalanced.
       Here every occurrence is tried from last to first, so a response whose final
       box was cut off by the token cap still scores on its last complete box
       instead of falling through to the numeric-tail heuristic.
    2. Upstream's ``remove_boxed`` asserts the prefix is exactly ``\\boxed{`` and
       therefore returns None for ``\\fbox{...}`` even though its own scanner
       accepts ``\\fbox``. Here both commands are handled the same way.
    3. An EMPTY box is skipped rather than returned. The task instruction itself ends
       with a literal ``\\boxed{}`` ("put your final answer within \\boxed{}"), so a
       response that echoes the prompt ends with an empty box; returning it would
       discard a real answer earlier in the same response. Guarding a hazard, not
       fixing an observed loss: on 60 MATH L3-5 prompts under Qwen2.5-Math-1.5B (base),
       which echoes the instruction on most prompts, this changed 0 of 60 -- the 20
       boxed-but-unextracted responses there contain ONLY the echoed empty box, with no
       real answer earlier to recover. Empty boxes carry no answer, so skipping them
       cannot lose one. See test group 10 for the case it does change.
    """
    starts: List[int] = [m.start() for m in BOXED_START_RE.finditer(text)]
    for idx in reversed(starts):
        content = _boxed_content_at(text, idx)
        if content is not None and content.strip():
            return content
    return None


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("$", "")
    ans = ans.rstrip(".")
    return ans.lower()


def extract_final_answer(text: str) -> ExtractionResult:
    boxed = _last_boxed(text)
    if boxed is not None and boxed.strip():
        return ExtractionResult(extracted=normalize_answer(boxed), used_boxed=True, success=True)

    m = FINAL_ANS_RE.search(text)
    if m:
        return ExtractionResult(extracted=normalize_answer(m.group(1)), used_boxed=False, success=True)

    n = NUMERIC_TAIL_RE.search(text.strip())
    if n:
        return ExtractionResult(extracted=normalize_answer(n.group(1)), used_boxed=False, success=True)

    return ExtractionResult(extracted=None, used_boxed=False, success=False)


def exact_match(pred_text: str, gt_answer: str) -> tuple[int, ExtractionResult]:
    pred = extract_final_answer(pred_text)
    if not pred.success:
        return 0, pred
    gt = normalize_answer(gt_answer)
    return int(pred.extracted == gt), pred
