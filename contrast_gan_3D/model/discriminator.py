from collections import OrderedDict

import torch
import torch.nn as nn

from contrast_gan_3D.model.blocks import ConvBlock3D


class PatchGAN(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        discriminator_depth: int,
        n_feature_maps: int,
        kernel_size: int = 4,
        padding: int = 1,
    ):
        super().__init__()

        model = [
            (
                "first",
                ConvBlock3D(
                    channels_in,
                    n_feature_maps,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    norm_layer=nn.Identity,
                    activation_fn=nn.LeakyReLU,
                    negative_slope=0.2,
                ),
            )
        ]
        # gradually increase the number of filters, capping at 8
        middle = []
        for n in range(discriminator_depth):
            in_ = min(2**n, 8) * n_feature_maps
            out_ = min(2 ** (n + 1), 8) * n_feature_maps
            middle.append(
                ConvBlock3D(
                    in_,
                    out_,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    activation_fn=nn.LeakyReLU,
                    negative_slope=0.2,
                )
            )
        model.append(("middle", nn.Sequential(*middle)))
        model.append(
            (
                "last",
                nn.Conv3d(
                    out_,
                    channels_out,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
            )
        )
        self.model = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
