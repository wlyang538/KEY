"""Equations (12)-(14). Average tokens within each item, then average items."""
import torch
import torch.nn.functional as F


def aligned(logits, batch):
    z = logits[:, :-1].float()
    labels = batch.labels[:, 1:]
    mask = labels.ne(-100) & batch.attention_mask[:, 1:].bool() & batch.attention_mask[:, :-1].bool()
    return z, labels.masked_fill(~mask, 0), mask


def item_mean(values, mask):
    return ((values.masked_fill(~mask, 0).sum(-1)) / mask.sum(-1)).mean()


def mean_answer_logit(logits, batch):
    z, labels, mask = aligned(logits, batch)
    return item_mean(z.gather(-1, labels.unsqueeze(-1)).squeeze(-1), mask)


def edit_loss(logits, batch):
    z, labels, mask = aligned(logits, batch)
    logp = z.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return item_mean(-logp, mask)


def forget_loss(logits, batch):
    z, labels, mask = aligned(logits, batch)
    # -log(1-p_y) = logsumexp(all logits) - logsumexp(non-target logits).
    # Stable even when softmax rounds p_y to 1; no gradient-killing clamp.
    other = z.scatter(-1, labels.unsqueeze(-1), float("-inf"))
    return item_mean(z.logsumexp(-1) - other.logsumexp(-1), mask)


def retain_loss(logits, reference_logits, batch):
    z, _, mask = aligned(logits, batch)
    reference, _, _ = aligned(reference_logits.detach(), batch)
    kl = F.kl_div(z.log_softmax(-1), reference.log_softmax(-1),
                  reduction="none", log_target=True).sum(-1)
    return edit_loss(logits, batch) + item_mean(kl, mask)
