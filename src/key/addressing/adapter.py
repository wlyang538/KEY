"""Differentiable low-rank residuals, training only alpha, with explicit merge."""
from contextlib import contextmanager
import torch
from torch import nn
from ..models.selection import resolve_modules


class SparseAddress(nn.Module):
    def __init__(self, dictionary, device):
        super().__init__()
        self.register_buffer("U", dictionary.U.detach().to(device=device, dtype=torch.float32).clone())
        self.register_buffer("V", dictionary.V.detach().to(device=device, dtype=torch.float32).clone())
        self.register_buffer("indices", dictionary.indices.to(device).clone())
        self.alpha = nn.Parameter(torch.zeros(len(dictionary.indices), device=device))

    def matrix(self):
        size = self.U.shape[1] * self.V.shape[1]
        return self.alpha.new_zeros(size).scatter(0, self.indices, self.alpha).view(
            self.U.shape[1], self.V.shape[1])

    def forward(self, x):
        return ((x.float() @ self.V) @ self.matrix().T) @ self.U.T

    def delta(self):
        return self.U @ self.matrix() @ self.V.T


class AddressedModel:
    """Owns temporary hooks and coefficient parameters, not a copy of the base LLM.

    Use as a context manager. close() removes residuals and restores original
    requires_grad/training flags. merge() permanently applies deltas and closes.
    No concurrent forwards on this model while the adapter is active.
    """
    def __init__(self, model, dictionaries):
        if getattr(model, "_key_active", False):
            raise ValueError("This model already has an active KEY adapter")
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("Disable gradient checkpointing for this implementation")
        self.model = model
        self.modules = resolve_modules(model, dictionaries.keys())
        for name, module in self.modules.items():
            d = dictionaries[name]
            if (d.U.shape[0], d.V.shape[0]) != tuple(module.weight.shape):
                raise ValueError(f"Dictionary/weight shape mismatch for {name}")
        self.addresses = nn.ModuleDict({str(i): SparseAddress(dictionaries[name], m.weight.device)
                                       for i, (name, m) in enumerate(self.modules.items())})
        self.names = list(self.modules)
        self._flags = {p: p.requires_grad for p in model.parameters()}
        self._modes = {m: m.training for m in model.modules()}
        self._handles = []
        self.enabled = True
        self.closed = False
        model._key_active = True
        try:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            for i, module in enumerate(self.modules.values()):
                address = self.addresses[str(i)]
                def hook(_module, inputs, output, address=address):
                    if not self.enabled:
                        return output
                    return output + address(inputs[0]).to(output.dtype)
                self._handles.append(module.register_forward_hook(hook))
        except Exception:
            self.close()
            raise

    def parameters(self):
        return self.addresses.parameters()

    def l1(self):
        return torch.stack([a.alpha.abs().sum() for a in self.addresses.values()]).sum()

    @contextmanager
    def disabled(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def close(self):
        if self.closed:
            return
        for h in self._handles:
            h.remove()
        for p, flag in self._flags.items():
            p.requires_grad_(flag)
        for m, mode in self._modes.items():
            m.training = mode
        self.model._key_active = False
        self.closed = True

    @torch.no_grad()
    def merge(self):
        if self.closed:
            raise RuntimeError("Cannot merge a closed/already merged adapter")
        # Validate before any mutation. Each delta is materialized only when needed.
        for a in self.addresses.values():
            if not torch.isfinite(a.alpha).all():
                raise ValueError("Cannot merge nonfinite coefficients")
        for i, module in enumerate(self.modules.values()):
            module.weight.add_(self.addresses[str(i)].delta().to(module.weight))
        self.close()
        return self.model

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
