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
    def __init__(self, bias: int, factor: int, min_HU: int, max_HU: int):
        super().__init__()
        self.min = torch.tensor((min_HU + bias) / factor)
        self.max = torch.tensor((max_HU + bias) / factor)

    def forward(self, source: Tensor, mask: torch.BoolTensor) -> Tensor:
        input_masked = torch.masked_select(source, mask)
        loss_min = torch.mean((torch.min(input_masked, self.min) - self.min) ** 2)
        loss_max = torch.mean((torch.max(input_masked, self.max) - self.max) ** 2)
        return loss_min + loss_max


class WassersteinLoss(nn.Module):
    @staticmethod
    def forward(input: Tensor, target: Optional[Tensor] = None) -> Tensor:
        ret = torch.mean(input)
        if target is not None:
            ret -= torch.mean(target)
        return ret
