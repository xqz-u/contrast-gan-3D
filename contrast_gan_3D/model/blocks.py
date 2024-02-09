import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        upsample: bool = False,
        output_padding: int = 0,
        padding_mode: str = "zeros",
        padding: int = 0,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        norm_layer: nn.Module = nn.BatchNorm3d,
        **activation_kwargs
    ):
        super().__init__()

        conv_class, args = nn.Conv3d, {}
        if upsample:
            conv_class, args = nn.ConvTranspose3d, {"output_padding": output_padding}

        self.conv = conv_class(
            channels_in,
            channels_out,
            kernel_size,
            stride=stride,
            bias=norm_layer not in [nn.BatchNorm3d, nn.GroupNorm],
            padding_mode=padding_mode,
            padding=padding,
            **args
        )
        self.normalization = norm_layer(channels_out)
        self.activation_fn = activation_fn(inplace=True, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(self.normalization(self.conv(x)))


# NOTE not customizing activation_fn and normalization_layer atm
class ResNetBlock3D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        n_feature_maps: int = 1,
        kernel_size: int = 3,
        dropout_prob: float = 0.0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        padding_amount = 1  # for residual connection to work
        self.conv0 = ConvBlock3D(
            channels_in,
            n_feature_maps,
            kernel_size,
            padding_mode=padding_mode,
            padding=padding_amount,
        )
        self.dropout = nn.Identity()
        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        self.conv1 = ConvBlock3D(
            n_feature_maps,
            channels_out,
            kernel_size,
            padding_mode=padding_mode,
            padding=padding_amount,
            activation_fn=nn.Identity,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv1(self.dropout(self.conv0(x)))  # with skip connection
