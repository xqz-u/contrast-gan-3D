from enum import Enum
from typing import Tuple, Union

import numpy as np
import torch

Shape3D = Tuple[int, int, int]
Array = Union[np.ndarray, torch.Tensor]


class ScanType(Enum):
    OPT = 0
    LOW = -1
    HIGH = 1
