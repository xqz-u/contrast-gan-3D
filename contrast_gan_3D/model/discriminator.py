from collections import OrderedDict

import torch
import torch.nn as nn

from contrast_gan_3D.model.blocks import ConvBlock3D


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        channels_in: int,
        init_channels_out: int,
        discriminator_depth: int,
        kernel_size: int = 4,
        padding: int = 1,
        norm_layer: nn.Module = nn.BatchNorm3d,
    ):
        super().__init__()

        stride, negative_slope = 2, 0.2
        model = [
            (
                "first",
                ConvBlock3D(
                    channels_in,
                    init_channels_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=nn.Identity,
                    activation_fn=nn.LeakyReLU,
                    negative_slope=negative_slope,
                ),
            )
        ]
        # gradually increase the number of filters, and decrease spatial extent:
        # the critic looks at increasingly finer feature maps
        middle = []
        for n in range(discriminator_depth):
            in_ = min(2**n, 8) * init_channels_out
            out_ = min(2 ** (n + 1), 8) * init_channels_out
            middle.append(
                ConvBlock3D(
                    in_,
                    out_,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=norm_layer,
                    activation_fn=nn.LeakyReLU,
                    negative_slope=negative_slope,
                )
            )
        model.append(("middle", nn.Sequential(*middle)))
        model.append(
            (
                "last",
                nn.Conv3d(
                    out_,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
            )
        )
        self.model = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
