import os
import random
from time import strftime
from typing import Any

import numpy as np
import torch


def object_name(el: object) -> str:
    return el.__class__.__name__


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def now_str() -> str:
    return strftime("%H:%M:%S")


def to_CPU(t) -> Any:
    if hasattr(t, "is_cuda") and t.is_cuda:
        t = t.detach().cpu()
    return t
