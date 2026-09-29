"""A0 HARD GATE: adapter isolation / per-request routing. Two distinct in-memory adapters in
one batch must each TRACK THEIR OWN adapter (not leak the other's). We assert routing via a
similarity matrix (batched[i] must be far more similar to solo[i] than to solo[j]) rather than
bitwise batched==solo, because vLLM's grouped multi-LoRA punica kernels differ numerically from
the single-adapter path — a real effect, not a leak. Run:
  CUDA_VISIBLE_DEVICES=0 $P4_PY track_a/tests/test_adapter_isolation.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/home/hyin66/.cache/hf_datasets")
os.environ.setdefault("XDG_CONFIG_HOME", "/home/hyin66/.cache/p4_xdg_config")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SIGMA = 0.05


def toks(o): return list(o.outputs[0].token_ids)
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
def agree(a, b):  # fraction of positions matching over min length
    m = min(len(a), len(b));  return (sum(1 for i in range(m) if a[i] == b[i]) / m) if m else 0.0


def main():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import lora_cfg

    tmpl = lora_cfg.build_template(MODEL)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85, max_model_len=1024,
              enforce_eager=True, enable_prefix_caching=False,
              enable_lora=True, max_loras=4, max_lora_rank=lora_cfg.LORA_RANK, max_cpu_loras=8,
              worker_extension_cls="es_bench.track_a.src.lora_es_worker.ESLoRAWorker")
    llm.collective_rpc("es_lora_init", args=(tmpl, lora_cfg.LORA_RANK, lora_cfg.LORA_ALPHA, lora_cfg.TARGET_MODULES, 0))
    idA, idB = llm.collective_rpc("es_lora_inject_members", args=(1, [11111, 22222], SIGMA))[0]
    a1 = LoRARequest("m_a", idA, "/es/inmem"); a2 = LoRARequest("m_b", idB, "/es/inmem")
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    P = "Q: What is the capital of France? A:"

    soloA = toks(llm.generate([P], sp, lora_request=a1)[0])
    soloB = toks(llm.generate([P], sp, lora_request=a2)[0])
    # batch: [A, B, B, A] -> different adapters, repeated, shuffled positions
    out = llm.generate([P, P, P, P], sp, lora_request=[a1, a2, a2, a1])
    bt = [toks(o) for o in out]
    which = [a1, a2, a2, a1]; own = [soloA, soloB, soloB, soloA]; other = [soloB, soloA, soloA, soloB]

    print(f"soloA==soloB? {soloA==soloB}  (must be False = adapters distinct)")
    fails = []
    def check(c, n): print(("  PASS " if c else "  FAIL ")+n); (fails.append(n) if not c else None)

    check(soloA != soloB, "adapters produce distinct outputs [else test vacuous]")
    # routing: each batched request tracks its OWN adapter far more than the other
    for i in range(4):
        ao, at = agree(bt[i], own[i]), agree(bt[i], other[i])
        print(f"  req{i} ({'A' if which[i] is a1 else 'B'}): agree(own)={ao:.2f} agree(other)={at:.2f}  lcp_own={lcp(bt[i],own[i])} lcp_other={lcp(bt[i],other[i])}")
        check(bt[i][0] == own[i][0], f"req{i} first token == own adapter")
        check(ao > at + 0.15, f"req{i} tracks OWN adapter (agree_own >> agree_other)")
    # two identical-adapter requests in the same batch must agree with each other
    check(bt[1] == bt[2], "req1==req2 (same adapter B, same prompt) consistent within batch")

    print(f"\n{'ALL PASS - routing/isolation confirmed' if not fails else 'FAILED: '+'; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
