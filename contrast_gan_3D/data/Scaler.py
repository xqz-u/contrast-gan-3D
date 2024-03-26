from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch import nn

from contrast_gan_3D.alias import Array
from contrast_gan_3D.data.utils import minmax_norm


class Scaler(ABC):
    @abstractmethod
    def __call__(self, x: Array) -> Array:
        ...

    @abstractmethod
    def unscale(self, x: Array) -> Array:
        ...


# pack low, high, shift in one place that still uses autograd
# NOTE does not inherit from `Scaler` but uses same interface
class MinMaxScaler(nn.Module):
    def __init__(self, low: float, high: float, b: float = 0):
        super().__init__()
        self.low = low
        self.high = high
        self.b = b

    def forward(self, x: Array):
        return minmax_norm(x, (self.low, self.high)) - self.b

    def unscale(self, x: Array) -> Array:
        return (x + self.b) * (self.high - self.low) + self.low


class FactorMinMaxScaler(MinMaxScaler):
    def __init__(self, low: float, high: float, factor: int, b: float = 0):
        super().__init__(low, high, b)
        self.factor = factor

    def forward(self, x: Array):
        return super().forward(x * self.factor)

    # actually unused
    def unscale(self, x: Array) -> Array:
        return super().unscale(x) / self.factor


@dataclass
class ZeroCenterScaler(Scaler):
    low: int
    high: int
    shift: int = field(init=False, default=None)

    def __post_init__(self):
        self.shift = (self.high - abs(self.low)) // 2

    def __call__(self, x: Array) -> Array:
        return x - self.shift

    def unscale(self, x: Array) -> Array:
        return x + self.shift


@dataclass
class FactorZeroCenterScaler(ZeroCenterScaler):
    factor: int

    def __call__(self, x: Array) -> Array:
        return super().__call__(x) / self.factor

    def unscale(self, x: Array) -> Array:
        return super().unscale(x * self.factor)
