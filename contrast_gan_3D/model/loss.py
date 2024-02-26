from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class StableStd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: Tensor) -> Tensor:
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        tensor = ctx.tensor.detach()
        assert tensor.numel() > 1
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
        assert source.shape == target.shape, "Input shapes are different"

        cc = ((source - source.mean()) * (target - target.mean())).mean()
        std = self.stablestd(source) * self.stablestd(target)
        return -(cc / (std + 1e-8))


class HULoss(nn.Module):
    def __init__(self, HU_diff: int, min_HU: int, max_HU: int):
        super().__init__()
        self.unscaled_min = min_HU
        self.unscaled_max = max_HU
        self.HU_diff = HU_diff

    def forward(
        self, batch: Tensor, mask: torch.BoolTensor, scans_min: Tensor
    ) -> Tensor:
        batch_flat = batch.reshape(len(batch), -1)

        min_ = ((self.unscaled_min + scans_min) / self.HU_diff)[:, None]
        max_ = ((self.unscaled_max + scans_min) / self.HU_diff)[:, None]

        loss_min = (torch.min(batch_flat, min_) - min_).reshape(batch.shape)
        loss_min = torch.mean(loss_min[mask] ** 2)

        loss_max = (torch.max(batch_flat, max_) - max_).reshape(batch.shape)
        loss_max = torch.mean(loss_max[mask] ** 2)

        return loss_min + loss_max


# NOTE this is not really a Wasserstein loss, it's adapted to be used with a
# PatchGAN. *TODO* rewrite to avoid confusion
class WassersteinLoss(nn.Module):
    @staticmethod
    def forward(fake: Tensor, real: Optional[Tensor] = None) -> Tensor:
        ret = torch.mean(fake)
        if real is not None:
            ret -= torch.mean(real)
        return ret
