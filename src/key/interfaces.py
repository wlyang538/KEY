"""Tensor contracts only: no tokenization, dataset readers, or preprocessing."""
from dataclasses import dataclass
from typing import Callable, Iterable
import torch


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self):
        if self.input_ids.ndim != 2 or self.input_ids.shape[1] < 2:
            raise ValueError("Expected [batch, sequence>=2] input_ids")
        if self.labels.shape != self.input_ids.shape or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("All batch tensors must have the same shape")
        if self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise TypeError("input_ids and labels must be int64")
        valid = self.labels[:, 1:].ne(-100) & self.attention_mask[:, 1:].bool()
        valid = valid & self.attention_mask[:, :-1].bool()
        if not valid.any(dim=1).all():
            raise ValueError("Every example needs at least one valid next-token answer label")

    def to(self, device):
        return TokenBatch(*(x.to(device) for x in (self.input_ids, self.attention_mask, self.labels)))


BatchFactory = Callable[[], Iterable[TokenBatch]]


def forward_logits(model, batch: TokenBatch):
    output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                   use_cache=False)
    return output if isinstance(output, torch.Tensor) else output.logits
