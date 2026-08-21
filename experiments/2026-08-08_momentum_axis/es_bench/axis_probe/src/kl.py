"""KL-to-base proxy: base-greedy NLL drift (fully in-process, no 2nd model).

Rationale: exact per-token full-vocab KL needs theta in HF form (vLLM fuses param
names -> un-fusing is model-specific/risky). Instead we measure, on a fixed held-out
prompt set, how much theta raises the NLL of the BASE model's own greedy continuation:

    D = mean_prompts mean_t ( logP_base(y_t | y_<t) - logP_theta(y_t | y_<t) )

where y = the base's greedy completion (captured once, at step 0, under the original
base weights). D >= ~0, grows monotonically with divergence for small drifts, and is
computed IDENTICALLY for binary and continuous arms -> the D_cont / D_binary ratio is a
valid consistent proxy for the pre-registered KL ratio gate (1.2x / 1.5x). Absolute
full-vocab KL is a deferred refinement (TODO in REPORT).

capture_base(): call with original base weights live (step 0), returns a frozen record.
drift(): call with theta live, returns D against that record.
"""
from __future__ import annotations
import numpy as np


def build_kl_prompts(tok, n, data_seed, levels, train_size):
    """Fixed held-out prompt set = the val split (disjoint from training)."""
    from es_bench import data_math
    _, val = data_math.make_split("math", train_size, max(n, 200), data_seed, levels=levels)
    qs = [v["question"] for v in val[:n]]
    return [data_math.build_prompt(tok, q) for q in qs]


def capture_base(llm, prompts, max_tokens=None):
    """Under the ORIGINAL base weights: greedy-decode each prompt, record its token
    ids and the base per-token logprobs. Returns the frozen record for drift()."""
    from vllm import SamplingParams
    import config as C
    if max_tokens is None:
        max_tokens = C.MAX_TOKENS
    # Needed for the base (non -Instruct) model: its eos is <|endoftext|> but the ChatML
    # prompt ends turns with <|im_end|>, so without this every capture runs to max_tokens.
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=0, stop=C.STOP)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    rec = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        base_lp = [lp[tid].logprob for tid, lp in zip(c.token_ids, c.logprobs)]
        rec.append({"prompt": p, "comp_ids": list(c.token_ids), "base_lp": base_lp})
    return rec


def _theta_logprobs(llm, tok, rec):
    """Under CURRENT (theta) weights: logprob of the base-greedy tokens, teacher-forced."""
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    reqs, spans = [], []
    for r in rec:
        prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        full = prompt_ids + r["comp_ids"]
        reqs.append(TokensPrompt(prompt_token_ids=full))
        spans.append((len(prompt_ids), len(full)))
    outs = llm.generate(reqs, sp, use_tqdm=False)
    per = []
    for out, (a, b), r in zip(outs, spans, rec):
        plp = out.prompt_logprobs
        full = tok(r["prompt"], add_special_tokens=False)["input_ids"] + r["comp_ids"]
        lps = []
        for pos in range(a, b):
            e = plp[pos] if pos < len(plp) else None
            if e and full[pos] in e:
                lps.append(e[full[pos]].logprob)
        per.append(lps)
    return per


def drift(llm, tok, rec):
    """Mean per-token NLL drift D = mean(base_lp - theta_lp) over all tokens."""
    theta = _theta_logprobs(llm, tok, rec)
    num = den = 0.0
    for r, tl in zip(rec, theta):
        for bl, thl in zip(r["base_lp"], tl):
            num += (bl - thl); den += 1
    return float(num / den) if den else 0.0
