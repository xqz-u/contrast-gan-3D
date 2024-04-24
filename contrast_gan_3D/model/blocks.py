from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        is_2D: bool,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        upsample: bool = False,
        output_padding: int = 0,
        padding_mode: str = "zeros",
        padding: int = 0,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        norm_layer: nn.Module | None = None,
        **activation_kwargs
    ):
        super().__init__()

        conv_class, args = nn.Conv2d if is_2D else nn.Conv3d, {}
        if upsample:
            args = {"output_padding": output_padding}
            conv_class = nn.ConvTranspose2d if is_2D else nn.ConvTranspose3d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d if is_2D else nn.BatchNorm3d

        self.conv = conv_class(
            channels_in,
            channels_out,
            kernel_size,
            stride=stride,
            bias=norm_layer == nn.Identity,
            padding_mode=padding_mode,
            padding=padding,
            **args
        )
        self.normalization = norm_layer(channels_out)
        self.activation_fn = activation_fn(inplace=True, **activation_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation_fn(self.normalization(self.conv(x)))


class ResNetBlock(nn.Module):
    def __init__(
        self,
        is_2D: bool,
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        dropout_prob: float = 0.0,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        padding_amount = 1  # one way to make residual connection to work
        self.block0 = ConvBlock(
            is_2D,
            channels_in,
            channels_out,
            kernel_size,
            padding_mode=padding_mode,
            padding=padding_amount,
            activation_fn=nn.Identity,
        )
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.block1 = ConvBlock(
            is_2D,
            channels_out,
            channels_out,
            kernel_size,
            padding_mode=padding_mode,
            padding=padding_amount,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block1(self.dropout(self.block0(x)))  # with skip connection
