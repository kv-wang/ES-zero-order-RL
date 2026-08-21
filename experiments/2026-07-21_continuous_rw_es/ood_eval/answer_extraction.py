import re
from dataclasses import dataclass
from typing import Optional

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANS_RE = re.compile(r"(?:final\s+answer\s*[:=]|answer\s*[:=])\s*([-+*/().,\w\\]+)", re.IGNORECASE)
NUMERIC_TAIL_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class ExtractionResult:
    extracted: Optional[str]
    used_boxed: bool
    success: bool


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(" ", "")
    ans = ans.replace("$", "")
    ans = ans.rstrip(".")
    return ans.lower()


def extract_final_answer(text: str) -> ExtractionResult:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return ExtractionResult(extracted=normalize_answer(boxed[-1]), used_boxed=True, success=True)

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
