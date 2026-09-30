"""Method orchestration; callers own model loading and tensor streams."""
from .config import DictionaryConfig, OptimizationConfig
from .models.selection import resolve_modules
from .gradients.factors import iter_factors
from .dictionary.construction import construct
from .dictionary.filtering import relevance, select_atoms
from .addressing.adapter import AddressedModel
from .addressing.trainer import train_address


def build_dictionary(model, module_names, pool_items, config=DictionaryConfig(), progress=None):
    if getattr(model, "_key_active", False):
        raise ValueError("Build dictionaries on the unmodified base model")
    modules = resolve_modules(model, module_names)
    result = {}
    # Process one matrix at a time to bound factor storage; pool factory is replayed.
    for name in modules:
        result[name] = construct(iter_factors(model, name, pool_items), config)
        if progress is not None:
            progress(name)
    return result


def filter_dictionary(model, dictionaries, target_items, retain_items=None,
                      config=DictionaryConfig()):
    if getattr(model, "_key_active", False):
        raise ValueError("Score dictionaries on the unmodified base model")
    resolve_modules(model, dictionaries)
    filtered, diagnostics = {}, {}
    for name, dictionary in dictionaries.items():
        target = relevance(dictionary, iter_factors(model, name, target_items))
        retained = relevance(dictionary, iter_factors(model, name, retain_items)) if retain_items is not None else None
        filtered[name], scores = select_atoms(dictionary, target, retained, config)
        diagnostics[name] = {"target": target, "retain": retained, "score": scores}
    return filtered, diagnostics


def optimize_address(model, dictionaries, target_batches, retain_batches=None,
                     config=OptimizationConfig(), callback=None):
    """Returns an active adapter and training history; caller must close or merge."""
    adapter = AddressedModel(model, dictionaries)
    try:
        history = train_address(adapter, target_batches, retain_batches, config, callback)
    except Exception:
        adapter.close()
        raise
    return adapter, history
