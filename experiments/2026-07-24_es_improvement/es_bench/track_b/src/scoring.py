"""Continuous-reward secondary key: teacher-forced gold-answer likelihood.

secondary = P-bar = mean_b ( mean_t P(y*_{b,t} | x_b + z_b + opener) )

  x_b     = prompt
  z_b     = member generation truncated at its final-answer span (house extractor
            precedence); if no answer found, z_b = full generation
  opener  = ANSWER_OPENER ("\\boxed{"), so we score the ANSWER CONTENT tokens, not
            the easy formatting tokens
  y*_b    = canonical gold answer (gt) tokens, teacher-forced
  P       = exp(logprob); "mean-per-token probability" = arithmetic mean over tokens

The whole B-batch is scored in ONE vLLM prompt_logprobs call (max_tokens=1). The
caller MUST invoke this while member i's perturbed weights are live (immediately
after member i's generate(), before the shift to member i+1).

Design note (V1, flagged in report): the exact gold rendering (prime the opener,
score canonical answer content) is a Phase-4 choice; the spec left it implicit.
"""
from __future__ import annotations
import importlib.util
import os

import numpy as np

# reuse the house extractor regexes read-only (no sys.path pollution)
_AE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "ood_eval", "answer_extraction.py")
_spec = importlib.util.spec_from_file_location("p4_answer_extraction", _AE_PATH)
_ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae)
BOXED_RE, FINAL_ANS_RE, NUMERIC_TAIL_RE = _ae.BOXED_RE, _ae.FINAL_ANS_RE, _ae.NUMERIC_TAIL_RE


def truncate_at_answer_span(raw: str) -> str:
    """z_b: member text up to (excluding) its final-answer span. House precedence."""
    idx = raw.rfind("\\boxed{")
    if idx != -1:
        return raw[:idx]
    m = FINAL_ANS_RE.search(raw)
    if m:
        return raw[:m.start()]
    stripped = raw.rstrip()
    n = NUMERIC_TAIL_RE.search(stripped)
    if n:
        return stripped[:n.start()]
    return raw  # no answer span found -> full generation


def _gold_positions(tok, prefix_str: str, gold_str: str):
    """Return (full_ids, gold_slice) where gold_slice indexes the gold tokens in
    full_ids, robust to BPE merges at the prefix/gold boundary (common-prefix)."""
    prefix_ids = tok(prefix_str, add_special_tokens=False)["input_ids"]
    full_ids = tok(prefix_str + gold_str, add_special_tokens=False)["input_ids"]
    # longest shared prefix length
    k = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        k += 1
    if k >= len(full_ids):          # gold added no new tokens (degenerate)
        k = max(0, len(full_ids) - 1)
    return full_ids, slice(k, len(full_ids))


def build_scoring_inputs(tok, prompts, gen_texts, gts, opener: str):
    """Per prompt: token ids for [x_b + z_b + opener + gold] and the gold slice."""
    token_prompts, gold_slices = [], []
    for x_b, gen, gt in zip(prompts, gen_texts, gts):
        z_b = truncate_at_answer_span(gen)
        prefix = x_b + z_b + opener
        full_ids, gsl = _gold_positions(tok, prefix, str(gt))
        token_prompts.append(full_ids)
        gold_slices.append(gsl)
    return token_prompts, gold_slices


def score_secondary(llm, tok, prompts, gen_texts, gts, opener="\\boxed{"):
    """One batched prompt_logprobs call under the CURRENT (member) weights.

    Returns (secondary_mean, per_prompt_pbar). per-token prob = exp(logprob),
    averaged within a prompt then across prompts.
    """
    from vllm import SamplingParams, TokensPrompt

    token_prompts, gold_slices = build_scoring_inputs(tok, prompts, gen_texts, gts, opener)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    reqs = [TokensPrompt(prompt_token_ids=ids) for ids in token_prompts]
    outs = llm.generate(reqs, sp, use_tqdm=False)

    per_prompt = []
    for out, ids, gsl in zip(outs, token_prompts, gold_slices):
        plp = out.prompt_logprobs  # list len == len(ids); [0] is None
        probs = []
        for pos in range(gsl.start, gsl.stop):
            entry = plp[pos] if pos < len(plp) else None
            if not entry:
                continue
            tid = ids[pos]
            lp = entry.get(tid)
            if lp is None:
                continue
            probs.append(float(np.exp(lp.logprob)))
        per_prompt.append(float(np.mean(probs)) if probs else 0.0)
    return float(np.mean(per_prompt)) if per_prompt else 0.0, per_prompt
