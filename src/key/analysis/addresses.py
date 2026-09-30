"""Address diagnostics. Only compare coefficients from identical bases/order."""
import torch
import torch.nn.functional as F


def concatenate_address(adapter):
    return torch.cat([a.matrix().detach().flatten().cpu() for a in adapter.addresses.values()])


def cosine_similarity(left, right, *, absolute=False):
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Expected aligned vectors of identical shape")
    if left.norm() == 0 or right.norm() == 0:
        raise ValueError("Cosine undefined for a zero address")
    value = F.cosine_similarity(left[None], right[None]).item()
    return abs(value) if absolute else value


def active_fraction(adapter, threshold=1e-5):
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    alpha = torch.cat([a.alpha.detach().cpu() for a in adapter.addresses.values()])
    return (alpha.abs() > threshold).float().mean().item()
