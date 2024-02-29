from enum import Enum
from typing import Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

Shape3D = Tuple[int, int, int]
Array = Union[np.ndarray, torch.Tensor]
BGenAugmenter = Union[
    NonDetMultiThreadedAugmenter, MultiThreadedAugmenter, SingleThreadedAugmenter
]


class ScanType(Enum):
    OPT = 0
    LOW = -1
    HIGH = 1
