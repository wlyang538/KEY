"""Knowledge-wise relevance, positive Q90 normalization and per-matrix top-k."""
import torch
from .atoms import AtomDictionary


def relevance(dictionary, factors):
    total = torch.zeros(dictionary.size)
    count = 0
    try:
        for x, g in factors:
            # Absolute value AFTER summing all token contributions for one item.
            projection = (g @ dictionary.U).T @ (x @ dictionary.V)
            total += projection.flatten().abs()
            count += 1
    finally:
        if hasattr(factors, "close"):
            factors.close()
    if count == 0:
        raise ValueError("Empty scoring stream; use None for absent retention")
    return total / count


def normalize_positive(scores, epsilon):
    positive = scores[scores > 0]
    q90 = torch.quantile(positive, 0.9) if positive.numel() else scores.new_tensor(0.)
    return scores / (q90 + epsilon)


def select_atoms(dictionary, target_scores, retain_scores, config):
    for value in (target_scores, retain_scores):
        if value is not None and (value.shape != (dictionary.size,) or
                                  not torch.isfinite(value).all() or (value < 0).any()):
            raise ValueError("Expected finite nonnegative relevance for every candidate atom")
    score = normalize_positive(target_scores, config.epsilon)
    if retain_scores is not None:
        score = score / (normalize_positive(retain_scores, config.epsilon) + config.epsilon)
    # Stable ties choose smaller global atom IDs. Preserve canonical global order.
    chosen = torch.argsort(score, descending=True, stable=True)[:min(config.atom_budget, dictionary.size)]
    return AtomDictionary(dictionary.U, dictionary.V, chosen.sort().values), score
