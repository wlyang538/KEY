"""Collect x and d s_i / d y without constructing a full weight gradient."""
import torch
from ..interfaces import forward_logits
from ..objectives.token_losses import mean_answer_logit


def iter_factors(model, module_name, batches):
    """One item per batch is required to preserve knowledge-wise averaging.

    Includes all valid sequence positions, including the prompt: answer scores
    can depend on prompt activations through attention. Zero-gradient rows can
    subsequently be dropped exactly.
    """
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("Disable gradient checkpointing for factor collection")
    module = model.get_submodule(module_name)
    modes = {m: m.training for m in model.modules()}
    flags = {p: p.requires_grad for p in model.parameters()}
    captured = []

    def hook(_module, inputs, output):
        captured.append((inputs[0], output))

    handle = module.register_forward_hook(hook)
    try:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        module.weight.requires_grad_(True)
        for batch in batches():
            if batch.input_ids.shape[0] != 1:
                raise ValueError("Dictionary/scoring factories must yield one knowledge item at a time")
            batch = batch.to(module.weight.device)
            captured.clear()
            with torch.enable_grad():
                logits = forward_logits(model, batch)
                if len(captured) != 1:
                    raise ValueError("Editable module must be called exactly once per forward")
                x, y = captured[0]
                if x.shape[:2] != batch.input_ids.shape or x.ndim != 3:
                    raise ValueError("Expected linear activations [batch, sequence, hidden]")
                score = mean_answer_logit(logits, batch)
                g, = torch.autograd.grad(score, y)
            valid = batch.attention_mask.bool()
            factors = (x.detach()[valid].float().cpu(), g.detach()[valid].float().cpu())
            captured.clear()
            del logits, score, x, y, g
            yield factors
    finally:
        handle.remove()
        for p, flag in flags.items():
            p.requires_grad_(flag)
        for m, training in modes.items():
            m.training = training
