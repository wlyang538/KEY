"""KEY: factorized dictionaries and task-specific sparse addresses."""
from .config import DictionaryConfig, OptimizationConfig
from .interfaces import TokenBatch
from .pipeline import build_dictionary, filter_dictionary, optimize_address
from .addressing.adapter import AddressedModel
from .io.checkpoint import save_dictionary, load_dictionary, save_address, load_address

__all__ = ["DictionaryConfig", "OptimizationConfig", "TokenBatch", "build_dictionary",
           "filter_dictionary", "optimize_address", "AddressedModel", "save_dictionary",
           "load_dictionary", "save_address", "load_address"]
