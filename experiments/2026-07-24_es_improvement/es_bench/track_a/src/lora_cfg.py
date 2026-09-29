"""LoRA-ES configuration + tensor-template derivation (Qwen2 family).

LoRA cfg is SHARED with LoRA-GRPO later: rank 16, alpha 32, targets = attn{q,k,v,o}+MLP.
sigma_adapter is ES's OWN perturbation scale in adapter space (NOT the full-param sigma) —
exposed here, tuned in A1. B init = 0 (PEFT default => initial adapter is a no-op).

vLLM expects adapter tensor keys of the form:
  base_model.model.<module_path>.lora_A.weight   shape (r, in_features)
  base_model.model.<module_path>.lora_B.weight   shape (out_features, r)
"""
from __future__ import annotations

LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SIGMA_ADAPTER = 0.01          # ES perturbation scale in adapter space (tuned in A1)

_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}


def build_template(model_name: str, rank: int = LORA_RANK, targets=TARGET_MODULES):
    """Return [{module_path, keyA, keyB, shapeA=(r,in), shapeB=(out,r)}] for every
    (layer, target module), derived from the HF config (no model load)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_name)
    H = c.hidden_size
    I = c.intermediate_size
    L = c.num_hidden_layers
    head_dim = getattr(c, "head_dim", H // c.num_attention_heads)
    Hkv = c.num_key_value_heads * head_dim
    Hq = c.num_attention_heads * head_dim
    io = {  # module -> (in_features, out_features)
        "q_proj": (H, Hq), "k_proj": (H, Hkv), "v_proj": (H, Hkv), "o_proj": (Hq, H),
        "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H),
    }
    tmpl = []
    for l in range(L):
        for m in targets:
            sub = "self_attn" if m in _ATTN else "mlp"
            path = f"model.layers.{l}.{sub}.{m}"
            fin, fout = io[m]
            tmpl.append({
                "module_path": path,
                "keyA": f"base_model.model.{path}.lora_A.weight",
                "keyB": f"base_model.model.{path}.lora_B.weight",
                "shapeA": (rank, fin), "shapeB": (fout, rank),
            })
    return tmpl
