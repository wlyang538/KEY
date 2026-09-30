"""Tensor-only artifacts: no model weights, examples, or datasets are saved."""
import torch
from ..dictionary.atoms import AtomDictionary
from ..addressing.adapter import AddressedModel


def _payload(dictionaries):
    return {name: {"U": d.U.detach().cpu(), "V": d.V.detach().cpu(),
                   "indices": d.indices.cpu()} for name, d in dictionaries.items()}


def _read(path, kind, base_model_id):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state.get("kind") != kind:
        raise ValueError("Unsupported KEY checkpoint")
    if state["base_model_id"] != base_model_id:
        raise ValueError("Checkpoint base_model_id differs from requested base model")
    return state


def save_dictionary(path, dictionaries, *, base_model_id):
    torch.save({"format_version": 1, "kind": "dictionary", "base_model_id": base_model_id,
                "dictionaries": _payload(dictionaries)}, path)


def load_dictionary(path, *, base_model_id):
    state = _read(path, "dictionary", base_model_id)
    return {name: AtomDictionary(**value) for name, value in state["dictionaries"].items()}


def save_address(path, adapter, *, base_model_id):
    if adapter.closed:
        raise RuntimeError("Save the address before closing or merging the adapter")
    dictionaries, coefficients = {}, {}
    for i, name in enumerate(adapter.names):
        a = adapter.addresses[str(i)]
        dictionaries[name] = {"U": a.U.detach().cpu(), "V": a.V.detach().cpu(), "indices": a.indices.cpu()}
        coefficients[name] = a.alpha.detach().cpu()
    torch.save({"format_version": 1, "kind": "address", "base_model_id": base_model_id,
                "dictionaries": dictionaries, "coefficients": coefficients}, path)


def load_address(path, model, *, base_model_id):
    """Attach to the SAME base checkpoint (not a previously merged model)."""
    state = _read(path, "address", base_model_id)
    dictionaries = {n: AtomDictionary(**v) for n, v in state["dictionaries"].items()}
    for name, d in dictionaries.items():
        alpha = state["coefficients"][name]
        if alpha.shape != d.indices.shape or not torch.isfinite(alpha).all():
            raise ValueError(f"Invalid coefficients for {name}")
    adapter = AddressedModel(model, dictionaries)
    try:
        with torch.no_grad():
            for i, name in enumerate(adapter.names):
                adapter.addresses[str(i)].alpha.copy_(state["coefficients"][name])
    except Exception:
        adapter.close()
        raise
    return adapter
