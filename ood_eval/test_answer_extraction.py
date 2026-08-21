#!/usr/bin/env python
"""Tests for ood_eval/answer_extraction.py, written with the 2026-08-09 \\boxed fix.

Run:  python ood_eval/test_answer_extraction.py

The regression these guard against: the old r"\\boxed\\{([^{}]+)\\}" matched only
brace-free contents, so every \\frac / \\sqrt / \\text answer fell through to the
numeric-tail heuristic and could never equal a ground truth containing a brace.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_extraction import (  # noqa: E402
    _last_boxed,
    extract_final_answer,
    exact_match,
    normalize_answer,
)

FAILS = []


def check(name, got, want):
    if got != want:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")


def main():
    # --- 1. the regression itself: nested braces must parse ---
    nested = [
        (r"\boxed{18}", "18"),
        (r"\boxed{-3}", "-3"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\sqrt{3}}", r"\sqrt{3}"),
        (r"\boxed{2\sqrt{5}}", r"2\sqrt{5}"),
        (r"\boxed{x^{2}}", "x^{2}"),
        (r"\boxed{\text{even}}", r"\text{even}"),
        (r"\boxed{\frac{\sqrt{2}}{2}}", r"\frac{\sqrt{2}}{2}"),   # two levels deep
    ]
    for text, want in nested:
        check(f"nested {text}", _last_boxed(text), want)

    # --- 2. last box wins, matching the previous findall()[-1] semantics ---
    check("last wins",
          _last_boxed(r"first \boxed{1} then \boxed{\frac{3}{4}}"), r"\frac{3}{4}")

    # --- 3. truncated final box falls back to the last COMPLETE one ---
    #     (deviation 1 from upstream, which would give up and return None)
    check("truncated tail",
          _last_boxed(r"\boxed{7} and then \boxed{\frac{1"), "7")
    check("only box truncated",
          _last_boxed(r"the answer is \boxed{\frac{1"), None)

    # --- 4. \fbox is accepted like \boxed (deviation 2 from upstream) ---
    check("fbox", _last_boxed(r"\fbox{42}"), "42")

    # --- 5. malformed / absent ---
    check("no braces", _last_boxed(r"\boxed 5"), None)
    check("no boxed", _last_boxed("the answer is 5"), None)
    check("empty box", extract_final_answer(r"\boxed{}").used_boxed, False)

    # --- 6. cascade order is unchanged ---
    r = extract_final_answer(r"blah \boxed{\frac{1}{2}} blah")
    check("tier1 used_boxed", (r.used_boxed, r.success, r.extracted),
          (True, True, r"\frac{1}{2}"))
    r = extract_final_answer("Final answer: 17")
    check("tier2", (r.used_boxed, r.success, r.extracted), (False, True, "17"))
    r = extract_final_answer("after much thought, 23")
    check("tier3 numeric tail", (r.used_boxed, r.success, r.extracted), (False, True, "23"))
    r = extract_final_answer("no answer here at all!")
    check("tier4 failure", (r.success, r.extracted), (False, None))

    # --- 7. end to end: a correct symbolic answer must now score 1 ---
    #     Before the fix this returned 0 for every gold containing a brace.
    gold = r"\frac{1}{2}"
    resp = r"We compute ... so the answer is \boxed{\frac{1}{2}}."
    score, res = exact_match(resp, gold)
    check("symbolic exact_match", score, 1)
    check("symbolic used_boxed", res.used_boxed, True)

    # a wrong symbolic answer must still score 0
    check("symbolic mismatch", exact_match(r"\boxed{\frac{1}{3}}", gold)[0], 0)

    # --- 8. DELIBERATE behaviour change, pinned here so it is not "fixed" by accident.
    # When a response boxes one thing and then writes "Final answer: <other>", the old
    # extractor's tier 1 failed on the nested box and fell through to tier 2, scoring the
    # trailing text. The fix makes tier 1 succeed, so the BOX now wins. If the gold happened
    # to match the trailing text, such a response used to score 1 by accident and now scores
    # 0. That is intended: a boxed answer is the model's answer. It also means the fix can
    # move individual questions in BOTH directions, which is why re-running evaluation is
    # required rather than assuming scores only improve.
    conflicting = r"\boxed{\frac{1}{2}}  Final answer: 0.5"
    r = extract_final_answer(conflicting)
    check("box shadows trailing text", (r.extracted, r.used_boxed), (r"\frac{1}{2}", True))
    check("box wins vs gold 0.5", exact_match(conflicting, "0.5")[0], 0)
    check("box wins vs gold 1/2", exact_match(conflicting, r"\frac{1}{2}")[0], 1)

    # --- 9. numeric behaviour is unchanged by the fix ---
    check("numeric still works", exact_match(r"\boxed{18}", "18")[0], 1)
    check("normalize spaces/$", normalize_answer(" $ 1/2 $. "), "1/2")

    # --- 10. an empty box never shadows a real answer ---
    # The task instruction ends with a literal \boxed{}, so any response that echoes the
    # prompt ends with an empty box. Before this was handled, _last_boxed returned "" and
    # any real answer earlier in the response was discarded. This is a guard, not a
    # measured recovery: on 60 MATH L3-5 prompts under Qwen2.5-Math-1.5B (base) it changed
    # 0 of 60, because there the echoed empty box is the ONLY box and no earlier answer
    # exists. It bites when a model answers and THEN echoes the instruction.
    echoed = r"The answer is \boxed{42}. Please put your final answer within \boxed{}."
    check("empty box skipped", extract_final_answer(echoed).extracted, "42")
    check("echoed prompt still scores", exact_match(echoed, "42")[0], 1)
    check("whitespace box skipped", extract_final_answer(r"\boxed{5} \boxed{   }").extracted, "5")
    # A response that is ONLY the echoed instruction has no answer and must stay unextracted.
    check("no answer stays None",
          extract_final_answer(r"Please put your final answer within \boxed{}.").extracted, None)

    print("ran 10 groups")
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all answer-extraction tests passed")


if __name__ == "__main__":
    main()
