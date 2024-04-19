from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

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
ArrayShape = Tuple[int, ...]
Array = Union[np.ndarray, torch.Tensor]
FoldType = List[Tuple[Union[str, Path], int]]
BGenAugmenter = Union[
    NonDetMultiThreadedAugmenter, MultiThreadedAugmenter, SingleThreadedAugmenter
]


class ScanType(Enum):
    OPT = 0
    LOW = -1
    HIGH = 1
