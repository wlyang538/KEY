"""Exact weighted factor stacking and truncated SVD, Eqs. (6)-(9)."""
import torch
from .atoms import AtomDictionary


def weighted_factors(x, g):
    nx, ng = x.norm(dim=-1), g.norm(dim=-1)
    valid = (nx > 0) & (ng > 0)
    rho_sqrt = (nx[valid] * ng[valid]).sqrt().unsqueeze(-1)
    return x[valid] / nx[valid, None] * rho_sqrt, g[valid] / ng[valid, None] * rho_sqrt


def construct(factors, config):
    if config.solver == "gram":
        return construct_gram(factors, config)
    a_rows, b_rows = [], []
    try:
        for x, g in factors:
            a, b = weighted_factors(x, g)
            if a.shape[0]:
                a_rows.append(a)
                b_rows.append(b)
    finally:
        if hasattr(factors, "close"):
            factors.close()
    if not a_rows:
        raise ValueError("No nonzero gradient factors in dictionary source")
    A, B = torch.cat(a_rows), torch.cat(b_rows)
    del a_rows, b_rows
    def basis(matrix, rank):
        if rank > min(matrix.shape):
            raise ValueError(f"rank={rank} exceeds available dimensions {tuple(matrix.shape)}")
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
        if singular[rank - 1] <= tolerance:
            raise ValueError("Requested rank exceeds numerical rank; lower the configured rank")
        return vh[:rank].T.contiguous()
    V = basis(A, config.rank_in)
    U = basis(B, config.rank_out)
    return AtomDictionary(U, V, torch.arange(config.rank_in * config.rank_out))


def construct_gram(factors, config):
    """Exact streaming A^T A / B^T B; eigenvectors equal right singular vectors.

    Uses FP64 accumulators, one editable module at a time. Memory is quadratic
    in hidden dimensions but independent of the number of knowledge tokens.
    """
    aa = bb = None
    try:
        for x, g in factors:
            a, b = weighted_factors(x.double(), g.double())
            if not len(a):
                continue
            if aa is None:
                aa = torch.zeros(a.shape[1], a.shape[1], dtype=torch.float64)
                bb = torch.zeros(b.shape[1], b.shape[1], dtype=torch.float64)
            aa.addmm_(a.T, a)
            bb.addmm_(b.T, b)
    finally:
        if hasattr(factors, "close"):
            factors.close()
    if aa is None:
        raise ValueError("No nonzero gradient factors in dictionary source")

    def basis(gram, rank):
        if rank > len(gram):
            raise ValueError("Requested rank exceeds matrix dimensions")
        values, vectors = torch.linalg.eigh(gram)
        tolerance = len(gram) * torch.finfo(gram.dtype).eps * values[-1].clamp_min(0)
        if values[-rank] <= tolerance:
            raise ValueError("Requested rank exceeds numerical rank; lower configured rank")
        return vectors[:, -rank:].flip(1).float().contiguous()
    V = basis(aa, config.rank_in)
    del aa
    U = basis(bb, config.rank_out)
    return AtomDictionary(U, V, torch.arange(config.rank_in * config.rank_out))
