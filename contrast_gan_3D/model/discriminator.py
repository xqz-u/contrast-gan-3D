from collections import OrderedDict

from torch import Tensor, nn

from contrast_gan_3D.model.blocks import ConvBlock
from contrast_gan_3D.model.utils import convolution_output_shape


class PatchGANDiscriminator(nn.Module):
    def __init__(
        self,
        channels_in: int,
        init_channels_out: int,
        discriminator_depth: int,
        is_2D: bool = False,
        kernel_size: int = 4,
        padding: int = 1,
        norm_layer: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__()

        stride = 2
        model = [
            (
                "first",
                ConvBlock(
                    is_2D,
                    channels_in,
                    init_channels_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=nn.Identity,
                    activation_fn=nn.LeakyReLU,
                    **kwargs,
                ),
            )
        ]
        # gradually increase the number of filters, and decrease spatial extent:
        # the critic looks at increasingly finer feature maps
        middle = []
        kwargs = kwargs.copy()
        if ps := kwargs.get("patch_size"):
            kwargs["patch_size"] = convolution_output_shape(
                ps, init_channels_out, kernel_size, padding, stride
            )
        for n in range(discriminator_depth):
            in_ = min(2**n, 8) * init_channels_out
            out_ = min(2 ** (n + 1), 8) * init_channels_out
            if ps := kwargs.get("patch_size"):
                kwargs["patch_size"] = convolution_output_shape(
                    ps, out_, kernel_size, padding, stride
                )
            middle.append(
                ConvBlock(
                    is_2D,
                    in_,
                    out_,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    norm_layer=norm_layer,
                    activation_fn=nn.LeakyReLU,
                    **kwargs,
                )
            )
        model.append(("middle", nn.Sequential(*middle)))
        model.append(
            (
                "last",
                (nn.Conv2d if is_2D else nn.Conv3d)(
                    out_,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
            )
        )
        self.model = nn.Sequential(OrderedDict(model))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
