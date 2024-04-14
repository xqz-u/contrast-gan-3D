from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from contrast_gan_3D.alias import Shape3D


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor) -> Tensor:
        # assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        tensor = ctx.tensor.detach()
        # assert tensor.numel() > 1
        result = ctx.result.detach()
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + 1e-6))
            * (tensor.detach() - tensor.mean().detach())
        )


class ZNCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stablestd = StableStd.apply

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        # assert source.shape == target.shape, "Input shapes are different"

        cc = ((source - source.mean()) * (target - target.mean())).mean()
        std = self.stablestd(source) * self.stablestd(target)
        return -(cc / (std + 1e-8))


class HULoss(nn.Module):
    def __init__(
        self: int,
        min_HU_contstraint: float,
        max_HU_constraint: float,
        patch_size: Shape3D,
    ):
        super().__init__()
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        device = torch.device(device)
        self.min_HU = torch.full(
            patch_size, min_HU_contstraint, dtype=torch.float32, device=device
        )
        self.max_HU = torch.full(
            patch_size, max_HU_constraint, dtype=torch.float32, device=device
        )

    def forward(self, batch: Tensor, mask: torch.BoolTensor) -> Tensor:
        lb, ub = torch.minimum(batch, self.min_HU), torch.maximum(batch, self.max_HU)
        loss_low = F.mse_loss(lb, self.min_HU, reduction="none")
        loss_high = F.mse_loss(ub, self.max_HU, reduction="none")
        loss = (loss_low + loss_high) * mask
        return loss.sum() / mask.sum()  # MSE over unmasked voxels


class WassersteinLoss(nn.Module):
    @staticmethod
    def forward(fake: Tensor, real: Optional[Tensor] = None) -> Tensor:
        ret = torch.mean(fake)
        if real is not None:
            ret -= torch.mean(real)
        return ret
