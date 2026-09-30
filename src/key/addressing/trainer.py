import torch
from ..interfaces import forward_logits
from ..objectives.token_losses import edit_loss, forget_loss, retain_loss


def repeating(factory):
    while True:
        seen = False
        for batch in factory():
            seen = True
            yield batch
        if not seen:
            raise ValueError("Batch factory returned an empty stream")


def train_address(adapter, target_batches, retain_batches, config, callback=None):
    if adapter.closed:
        raise RuntimeError("Adapter has already been closed")
    torch.manual_seed(config.seed)
    device = next(adapter.parameters()).device
    params = list(adapter.parameters())
    optimizer_cls = torch.optim.Adam if config.task == "edit" else torch.optim.AdamW
    optimizer = optimizer_cls(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    target_stream = repeating(target_batches)
    retain_stream = repeating(retain_batches) if retain_batches is not None else None
    lambda_ret = config.lambda_retain if retain_stream is not None else 0.0
    lambda_sel = config.lambda_select if retain_stream is not None else 1.0
    objective = edit_loss if config.task == "edit" else forget_loss
    # Hooks compute residuals and losses in FP32. Scale backward through FP16 base.
    use_scaler = device.type == "cuda" and any(p.dtype == torch.float16 for p in adapter.model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    history = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        target = next(target_stream).to(device)
        upd = objective(forward_logits(adapter.model, target), target)
        if not torch.isfinite(upd):
            raise FloatingPointError("Nonfinite update loss")
        # Separate backwards reduce simultaneous retention/target graph memory.
        scaler.scale(upd).backward()
        ret_value = 0.0
        if retain_stream is not None and lambda_ret > 0:
            retained = next(retain_stream).to(device)
            with adapter.disabled(), torch.no_grad():
                reference = forward_logits(adapter.model, retained).detach()
            ret = retain_loss(forward_logits(adapter.model, retained), reference, retained)
            if not torch.isfinite(ret):
                raise FloatingPointError("Nonfinite retention loss")
            scaler.scale(lambda_ret * ret).backward()
            ret_value = ret.detach().item()
            del reference, ret
        select = adapter.l1()
        scaler.scale(lambda_sel * select).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, config.clip_norm)
        finite = bool(torch.isfinite(norm))
        if not finite and not use_scaler:
            raise FloatingPointError("Nonfinite coefficient gradients")
        scaler.step(optimizer)  # Automatically skips overflowed FP16 iterations.
        scaler.update()
        record = {"step": step + 1, "update_loss": upd.detach().item(),
                  "retain_loss": ret_value, "l1": select.detach().item(),
                  "grad_norm": norm.item(), "optimizer_step_applied": finite}
        record["loss"] = record["update_loss"] + lambda_ret * ret_value + lambda_sel * record["l1"]
        history.append(record)
        if callback is not None:
            callback(record)
    return history
