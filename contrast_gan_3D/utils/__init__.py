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


def set_GPU(device_idx: int) -> torch.device:
    device_str = "cpu"
    if device_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_idx)
        device_str = f"cuda:{torch.cuda.current_device()}"
    return torch.device(device_str)


def now_str() -> str:
    return strftime("%H:%M:%S")


def to_CPU(t) -> Any:
    if hasattr(t, "is_cuda") and t.is_cuda:
        t = t.detach().cpu()
    return t
