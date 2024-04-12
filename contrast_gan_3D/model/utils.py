from typing import List, Optional, Union

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


def print_convolution_filters_shape(model: nn.Module, input_shape: torch.Tensor):
    print(f"Input shape: {list(input_shape)}")
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            kwargs = {}
            if isinstance(m, nn.ConvTranspose3d):
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
            print(
                f"{n:<40} -> {str(input_shape):<22} {params_str:<20} weight: {str(list(m.weight.shape)):<20}{bias_str}"
            )


def count_parameters(model: nn.Module, print:bool=False) -> int:
    tot = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            tot += p.numel()
            if print:
                print(n, p.numel())
    return tot
