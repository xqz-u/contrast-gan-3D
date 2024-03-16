import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from sklearn.model_selection import StratifiedKFold

from contrast_gan_3D.alias import BGenAugmenter, Shape3D
from contrast_gan_3D.constants import DEFAULT_SEED, MAX_HU, MIN_HU
from contrast_gan_3D.data.CCTADataLoader3D import CCTADataLoader3D
from contrast_gan_3D.utils import object_name


def cval_paths(
    n_folds: int,
    *dataset_paths: Iterable[Union[Path, str]],
    seed: Optional[int] = None,
) -> Tuple[List[List[Tuple[Union[str, Path], int]]], ...]:
    X, Y, train, val = [], [], [], []

    for df_path in dataset_paths:
        df = pd.read_excel(df_path)
        X += df["path"].values.tolist()
        Y += df["label"].values.tolist()
    X, Y = np.array(X), np.array(Y)

    cval = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in cval.split(X, Y):
        train.append(list(zip(X[train_idx], Y[train_idx])))
        val.append(list(zip(X[val_idx], Y[val_idx])))

    return train, val


def divide_scans_in_fold(
    fold: List[Tuple[Union[str, Path], int]]
) -> Dict[Any, List[Union[str, Path]]]:
    ret = defaultdict(list)
    for path, label in fold:
        ret[label].append(path)
    return ret


def create_dataloaders(
    train_fold: List[Tuple[str, int]],
    val_fold: List[Tuple[str, int]],
    train_mean: float,
    train_patch_size: Shape3D,
    val_patch_size: Union[Shape3D, int],
    train_batch_size: int,
    val_batch_size: int,
    normalize_range: Optional[Tuple[int, int]] = (MIN_HU, MAX_HU),
    num_workers: Tuple[int, int] = (1, 1),
    train_transform: Optional[Callable[[dict], dict]] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[int, BGenAugmenter], Dict[int, BGenAugmenter]]:
    pin_memory = torch.cuda.is_available()

    train_by_lab = divide_scans_in_fold(train_fold)
    train_loaders = {
        label: NonDetMultiThreadedAugmenter(
            CCTADataLoader3D(
                paths,
                train_patch_size,
                train_batch_size,
                normalize_range=normalize_range,
                dataset_mean=train_mean,
                shuffle=True,
                num_threads_in_multithreaded=num_workers[0],
                seed_for_shuffle=seed,
            ),
            train_transform,
            num_workers[0],
            pin_memory=pin_memory,
        )
        for label, paths in train_by_lab.items()
    }
    val_by_lab = divide_scans_in_fold(val_fold)

    val_loaders = {
        label: NonDetMultiThreadedAugmenter(
            CCTADataLoader3D(
                paths,
                val_patch_size,
                val_batch_size,
                normalize_range=normalize_range,
                dataset_mean=train_mean,
                shuffle=True,
                num_threads_in_multithreaded=num_workers[1],
                seed_for_shuffle=seed,
            ),
            Compose(
                [
                    NumpyToTensor(keys=["data"]),
                    NumpyToTensor(keys=["seg"], cast_to="bool"),
                ]
            ),
            num_workers[1],
            pin_memory=pin_memory,
        )
        for label, paths in val_by_lab.items()
    }
    return train_loaders, val_loaders


# author: ChatGPT
def global_overrides(config_path: Path):
    # Check if the path exists
    if not os.path.exists(config_path):
        print("Error: Config file does not exist.")
        return

    # Get the directory and file name from the path
    config_dir, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)

    # Add the directory to the sys.path if not already there
    if config_dir not in sys.path:
        sys.path.append(config_dir)

    # Use importlib to load the module
    return importlib.import_module(config_name)


def update_experiment_config(vars: dict) -> dict:
    return (
        {
            k: vars[k]
            for k in [
                "lr",
                "betas",
                "milestones",
                "lr_gamma",
                "max_HU_delta",
                "desired_HU_bounds",
                "HU_normalize_range",
                "generator_args",
                "discriminator_args",
                "train_patch_size",
                "train_batch_size",
                "val_patch_size",
                "val_batch_size",
                "dataset_paths",
                "train_transform_args",
                "train_iterations",
                "val_iterations",
                "train_generator_every",
                "seed",
                "cval_folds",
                "checkpoint_every",
                "validate_every",
                "log_every",
                "log_images_every",
                "num_workers",
            ]
        }
        | {
            k: object_name(vars[k])
            for k in [
                "generator",
                "generator_optim",
                "generator_lr_scheduler",
                "discriminator",
                "discriminator_optim",
                "discriminator_lr_scheduler",
            ]
        }
        | {"train_transform": str(vars["train_transform"])}
    )
