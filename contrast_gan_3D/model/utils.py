from typing import List, Union

import torch
from torch import Tensor, nn
from torch.autograd import grad


# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py
def wgan_gradient_penalty(
    real_batch: Tensor,
    fake_batch: Tensor,
    critic: nn.Module,
    device: Union[torch.device, str] = "cpu",
    lambda_: float = 10,
) -> Tensor:
    bs, *rest = real_batch.shape
    eps = torch.rand((bs,) + (1,) * len(rest), device=device).expand_as(real_batch)
    interpolation = eps * real_batch + (1 - eps) * fake_batch
    critic_logits = critic(interpolation)
    # https://discuss.pytorch.org/t/when-do-i-use-create-graph-in-autograd-grad/32853
    # retain_graph must also be True otherwise the computational graph of ``
    # freed before they can later be used in the full discriminator loss
    gradients, *_ = grad(
        outputs=critic_logits,
        inputs=interpolation,
        grad_outputs=torch.ones_like(critic_logits),
        create_graph=True,
    )
    gradients_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=-1)
    return lambda_ * (gradients_norm - 1).square().mean()


# simplified version of
# https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#torch.nn.Conv3d
def convolution_output_shape(
    dims: List[int],
    c_out: int,
    kernel_size: int,
    padding: int,
    stride: int,
    dilation: int = 1,
) -> List[int]:
    formula = lambda x: int(
        (x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    return [c_out] + [formula(d) for d in dims[1:]]  # assumes first dim is channels_in


def count_parameters(model: nn.Module) -> int:
    # tot = 0
    # for n, p in model.named_parameters():
    #     print(n)
    #     if p.requires_grad:
    #         tot += p.numel()
    #         print(p.numel())
    # return tot
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
