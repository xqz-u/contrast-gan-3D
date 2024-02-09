from typing import OrderedDict

import torch
import torch.nn as nn

from contrast_gan_3D.model.blocks import ConvBlock3D, ResNetBlock3D


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
        n_updown_sampling_blocks: int,
        n_feature_maps: int,
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
        model = [("first", ConvBlock3D(1, n_feature_maps, **first_and_last_common))]

        downsampling = []
        for i in range(n_updown_sampling_blocks):
            dim_in = n_feature_maps * 2**i
            dim_out = dim_in * 2
            downsampling.append(
                ConvBlock3D(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
            )
        model.append(("downsampling", nn.Sequential(*downsampling)))

        resnet_blocks = [
            ResNetBlock3D(
                dim_out,
                dim_out,
                dropout_prob=resnet_dropout_prob,
                padding_mode=resnet_padding_mode,
            )
        ] * n_resnet_blocks
        model.append(("resnet", nn.Sequential(*resnet_blocks)))

        upsampling = []
        for i in range(n_updown_sampling_blocks, 0, -1):
            dim_in = n_feature_maps * 2**i
            dim_out = int(dim_in / 2)
            upsampling.append(
                ConvBlock3D(
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
        model.append(
            (
                "last_conv",
                nn.Conv3d(n_feature_maps, 1, **first_and_last_common, bias=True),
            )
        )
        model.append(("tanh", nn.Tanh()))

        self.model = nn.Sequential(OrderedDict(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
