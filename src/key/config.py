from dataclasses import dataclass


@dataclass(frozen=True)
class DictionaryConfig:
    rank_in: int = 48
    rank_out: int = 48
    atom_budget: int = 1024
    epsilon: float = 1e-8
    solver: str = "gram"

    def __post_init__(self):
        if min(self.rank_in, self.rank_out, self.atom_budget) < 1 or self.epsilon <= 0:
            raise ValueError("Ranks, atom budget and epsilon must be positive")
        if self.solver not in {"gram", "svd"}:
            raise ValueError("solver must be gram or svd")


@dataclass(frozen=True)
class OptimizationConfig:
    task: str = "edit"
    steps: int = 200
    learning_rate: float = 0.03
    lambda_retain: float = 0.6
    lambda_select: float = 0.4
    clip_norm: float = 1.0
    weight_decay: float = 0.0  # Unspecified by paper; explicit engineering choice.
    seed: int = 2026

    def __post_init__(self):
        if self.task not in {"edit", "forget"}:
            raise ValueError("task must be edit or forget")
        if self.steps < 1 or self.learning_rate <= 0 or self.clip_norm <= 0:
            raise ValueError("steps, learning_rate and clip_norm must be positive")
        if min(self.lambda_retain, self.lambda_select, self.weight_decay) < 0:
            raise ValueError("Regularization weights must be nonnegative")
        if abs(self.lambda_retain + self.lambda_select - 1.0) > 1e-6:
            raise ValueError("Paper requires lambda_retain + lambda_select = 1")
