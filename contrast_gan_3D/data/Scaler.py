from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from contrast_gan_3D.alias import Array


class Scaler(ABC):
    @abstractmethod
    def __call__(self, x: Array) -> Array:
        ...

    @abstractmethod
    def unscale(self, x: Array) -> Array:
        ...


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
