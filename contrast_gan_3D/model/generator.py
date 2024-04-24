from typing import OrderedDict

import torch
import torch.nn as nn

from contrast_gan_3D.model.blocks import ConvBlock, ResNetBlock


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks between a few
    downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer
    project (https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        n_resnet_blocks: int,
        n_updownsample_blocks: int,
        init_channels_out: int,
        is_2D: bool = False,
        resnet_dropout_prob: float = 0.0,
        resnet_padding_mode: str = "zeros",
    ):
        assert n_resnet_blocks > 0

        super().__init__()

        first_and_last_common = {
            "kernel_size": 7,
            "padding_mode": "reflect",
            "padding": 3,
        }
        model = [
            ("first", ConvBlock(is_2D, 1, init_channels_out, **first_and_last_common))
        ]

        downsampling = []
        for i in range(n_updownsample_blocks):
            dim_in = init_channels_out * 2**i
            dim_out = dim_in * 2
            downsampling.append(
                ConvBlock(is_2D, dim_in, dim_out, kernel_size=3, stride=2, padding=1)
            )
        model.append(("downsampling", nn.Sequential(*downsampling)))

        resnet_blocks = [
            ResNetBlock(
                is_2D,
                dim_out,
                dim_out,
                dropout_prob=resnet_dropout_prob,
                padding_mode=resnet_padding_mode,
            )
            for _ in range(n_resnet_blocks)
        ]
        model.append(("resnet_backbone", nn.Sequential(*resnet_blocks)))

        upsampling = []
        for i in range(n_updownsample_blocks, 0, -1):
            dim_in = init_channels_out * 2**i
            dim_out = int(dim_in / 2)
            upsampling.append(
                ConvBlock(
                    is_2D,
                    dim_in,
                    dim_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    upsample=True,
                )
            )
        model.append(("upsampling", nn.Sequential(*upsampling)))
        last_conv = nn.Conv2d if is_2D else nn.Conv3d
        model.append(
            (
                "last_conv",
                last_conv(init_channels_out, 1, **first_and_last_common, bias=True),
            )
        )
        model.append(("tanh", nn.Tanh()))

        self.model = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
