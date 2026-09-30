from dataclasses import dataclass
import torch


@dataclass
class AtomDictionary:
    """D_m = U[:,p] V[:,q]^T; zero-based global ID m = p*rank_in + q."""
    U: torch.Tensor
    V: torch.Tensor
    indices: torch.Tensor

    def __post_init__(self):
        if self.U.ndim != 2 or self.V.ndim != 2 or self.indices.ndim != 1:
            raise ValueError("Expected matrix bases and vector indices")
        if self.indices.dtype != torch.long or self.indices.numel() == 0:
            raise ValueError("indices must be a nonempty int64 vector")
        if self.indices.unique().numel() != self.indices.numel():
            raise ValueError("Duplicate atom indices")
        if self.indices.min() < 0 or self.indices.max() >= self.size:
            raise ValueError("Atom index outside candidate dictionary")
        for basis in (self.U, self.V):
            if not torch.isfinite(basis).all():
                raise ValueError("Nonfinite basis")
            if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], device=basis.device,
                                  dtype=basis.dtype), atol=2e-4, rtol=2e-4):
                raise ValueError("Dictionary basis must be orthonormal")

    @property
    def size(self):
        return self.U.shape[1] * self.V.shape[1]

    def coefficient_matrix(self, alpha):
        return alpha.new_zeros(self.size).scatter(0, self.indices.to(alpha.device), alpha).view(
            self.U.shape[1], self.V.shape[1])

    def delta(self, alpha):
        return self.U.to(alpha) @ self.coefficient_matrix(alpha) @ self.V.to(alpha).T

    def full_address(self, alpha):
        return self.coefficient_matrix(alpha).flatten()
