"""Explicit editable linear modules; embedding and output head are not inferred."""
from torch import nn


def resolve_modules(model, names):
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide nonempty unique module names")
    found = {}
    seen_weights = set()
    for name in names:
        module = model.get_submodule(name)
        if type(module) is not nn.Linear:
            raise TypeError(f"{name}: only ordinary torch.nn.Linear is supported")
        if id(module.weight) in seen_weights:
            raise ValueError("Tied editable weights are unsupported")
        seen_weights.add(id(module.weight))
        found[name] = module
    if len({m.weight.device for m in found.values()}) != 1:
        raise ValueError("Use one device per model; device_map sharding is unsupported")
    return found


def llama_modules(model):
    suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    return [f"model.layers.{i}.{s}" for i in range(len(model.model.layers)) for s in suffixes]


def gptj_modules(model):
    suffixes = ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.fc_in", "mlp.fc_out")
    return [f"transformer.h.{i}.{s}" for i in range(len(model.transformer.h)) for s in suffixes]
