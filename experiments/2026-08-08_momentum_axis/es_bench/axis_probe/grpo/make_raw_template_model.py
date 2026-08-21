#!/usr/bin/env python
"""Build a local model dir whose chat template is the IDENTITY (concatenate message contents
verbatim, no <|im_start|> wrapping, no generation prompt).

WHY. verl applies tokenizer.apply_chat_template on every code path; countdown's protocol is a
raw completion prompt (the dataset `context`, ends mid-<think>). Pointing verl at this copy
makes its rollout prompt char-identical to the ES arms and to eval_core._eval_countdown,
without touching verl internals.

Weights/config are symlinked from the HF snapshot (no copy); only tokenizer_config.json is
rewritten. The ORIGINAL tokenizer_config.json is kept alongside as
tokenizer_config.orig.json -- the GRPO chain must restore it into hf_merged BEFORE the eval
battery, or the math-set evals (which need the real Qwen chat template) are silently broken.
"""
import json, os, shutil, sys

IDENTITY_TEMPLATE = "{%- for message in messages -%}{{ message['content'] }}{%- endfor -%}"


def main(model_id, out_dir):
    from huggingface_hub import snapshot_download
    src = snapshot_download(model_id)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(src):
        dst = os.path.join(out_dir, name)
        if name == "tokenizer_config.json" or os.path.lexists(dst):
            continue
        os.symlink(os.path.realpath(os.path.join(src, name)), dst)
    cfg = json.load(open(os.path.join(src, "tokenizer_config.json")))
    shutil.copy(os.path.join(src, "tokenizer_config.json"),
                os.path.join(out_dir, "tokenizer_config.orig.json"))
    cfg["chat_template"] = IDENTITY_TEMPLATE
    json.dump(cfg, open(os.path.join(out_dir, "tokenizer_config.json"), "w"), indent=2)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out_dir)
    probe = [{"role": "user", "content": "RAW_PROBE_123 <think>"}]
    got = tok.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    assert got == "RAW_PROBE_123 <think>", f"identity template failed: {got!r}"
    print(f"identity-template model ready at {out_dir} (weights symlinked from {src})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/qwen3b_rawtmpl")
