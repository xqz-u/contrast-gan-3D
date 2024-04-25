from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import grad

from contrast_gan_3D.alias import ArrayShape


# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/4.%20WGAN-GP/utils.py
def wgan_gradient_penalty(
    real_batch: Tensor,
    fake_batch: Tensor,
    critic: nn.Module,
    device: Union[torch.device, str] = "cpu",
    lambda_: float = 10,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    interp_sample_size, *t_shape = real_batch.shape
    if len(real_batch) != len(fake_batch):
        interp_sample_size = min(len(real_batch), len(fake_batch))
        rng = rng or np.random.default_rng()
        real_batch = real_batch[rng.integers(len(real_batch), size=interp_sample_size)]
        fake_batch = fake_batch[rng.integers(len(fake_batch), size=interp_sample_size)]
    eps = torch.rand((interp_sample_size,) + (1,) * len(t_shape), device=device)
    eps = eps.expand_as(real_batch)
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


# simplified versions of
# https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
def convolution_output_shape(
    dims: List[int],
    c_out: int,
    kernel_size: int,
    padding: int,
    stride: int,
    dilation: int = 1,
    transpose_output_padding: Optional[int] = None,
) -> List[int]:
    formula = lambda x: int(
        (x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    formula_transp = lambda x: int(
        (x - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + transpose_output_padding
        + 1
    )
    f = formula_transp if transpose_output_padding is not None else formula
    return [c_out] + [f(d) for d in dims[1:]]  # assumes first dim is channels_in


def compute_convolution_filters_shape(
    model: nn.Module, input_shape: ArrayShape, show: bool = True
) -> List[int]:
    printables = [f"Input shape: {list(input_shape)}"]
    for n, m in model.named_modules():
        if type(m) in (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d):
            kwargs = {}
            if isinstance(m, (nn.ConvTranspose3d, nn.ConvTranspose2d)):
                kwargs = {"transpose_output_padding": m.output_padding[0]}
            input_shape = convolution_output_shape(
                input_shape,
                m.out_channels,
                m.kernel_size[0],
                m.padding[0],
                m.stride[0],
                **kwargs,
            )
            bias_str = "" if m.bias is None else f" bias: {str(list(m.bias.shape))}"
            params_str = f"# params: {count_parameters(m)}"
            printables.append(
                f"{n:<40} -> {str(input_shape):<22} {params_str:<20} weight: {str(list(m.weight.shape)):<20}{bias_str}"
            )
    if show:
        for p in printables:
            print(p)
    return input_shape  # return the final output shape


def count_parameters(model: nn.Module, print: bool = False) -> int:
    tot = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            tot += p.numel()
            if print:
                print(n, p.numel())
    return tot
