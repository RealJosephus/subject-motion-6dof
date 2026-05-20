from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_motion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    target = target.to(pred)
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none")
    elif loss_type == "mse":
        loss = F.mse_loss(pred, target, reduction="none")
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    valid = mask.expand_as(loss)
    return (loss * valid).sum() / valid.sum().clamp_min(1)
