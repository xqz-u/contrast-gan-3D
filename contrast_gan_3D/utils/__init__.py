import multiprocessing as mp
import os
import random
from time import strftime
from typing import Any

import numpy as np
import torch

from contrast_gan_3D.alias import Array, Shape3D


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_GPU(device_idx: int) -> torch.device:
    # https://discuss.pytorch.org/t/gpu-device-ordering/60785/2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_str = "cpu"
    if device_idx is not None and torch.cuda.is_available():
        torch.cuda.set_device(device_idx)
        device_str = f"cuda:{torch.cuda.current_device()}"
    return torch.device(device_str)


# author: ChatGPT
def set_multiprocessing_start_method(method: str):
    try:
        mp.set_start_method(method)
    except RuntimeError as e:
        if "context has already been set" in str(e):
            print(f"Start method {method!r} has already been set.")
        else:
            raise


def now_str() -> str:
    return strftime("%H:%M:%S")


def to_CPU(t) -> Any:
    if hasattr(t, "is_cuda") and t.is_cuda:
        t = t.detach().cpu()
    return t


def parse_patch_size(target_shape: Shape3D, input_shape: Shape3D) -> np.ndarray:
    target_shape = np.array(target_shape)
    for i, dim in enumerate(target_shape):
        if dim == -1:
            target_shape[i] = input_shape[i]
    return target_shape


def swap_last_dim(t: Array) -> Array:
    *rest_dim, last_dim = np.arange(len(t.shape))
    if isinstance(t, np.ndarray):
        t = t.transpose(last_dim, *rest_dim)
    if isinstance(t, torch.Tensor):
        t = t.permute((last_dim, *rest_dim))
    return t


def downsample(a: np.ndarray, size: int) -> np.ndarray:
    return np.random.choice(a, size=size, replace=False)
