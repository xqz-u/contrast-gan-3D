import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.constants import DEFAULT_SEED
from contrast_gan_3D.data.CCTADataset import CCTADataset
from contrast_gan_3D.trainer.Reloader import Reloader
from contrast_gan_3D.utils import object_name


def crossval_paths(
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


def create_train_folds(
    train_patch_size: Shape3D,
    val_patch_size: Union[Shape3D, int],
    train_batch_size: int,
    val_batch_size: int,
    device_type: str,
    *dataset_paths: Iterable[Union[str, Path]],
    num_workers: Tuple[int, int] = (0, 0),
    max_HU_diff: Optional[int] = None,
    train_transform: Optional[Callable[[dict], dict]] = None,
    seed: int = DEFAULT_SEED,
    n_folds: int = 5,
) -> List[Tuple[Dict[int, Reloader]]]:
    train_folds, val_folds = crossval_paths(n_folds, *dataset_paths, seed=seed)

    train_by_lab = [divide_scans_in_fold(f) for f in train_folds]
    val_by_lab = [divide_scans_in_fold(f) for f in val_folds]

    rng, ret = np.random.default_rng(seed=seed), []
    # every i'th entry is a train/val fold
    for train, val in zip(train_by_lab, val_by_lab):
        train_fold = {
            label: Reloader(
                CCTADataset(
                    paths,
                    [label] * len(paths),
                    train_patch_size,
                    max_HU_diff=max_HU_diff,
                    rng=rng,
                    transform=train_transform,
                ),
                device_type,
                infinite=True,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers = num_workers[0]
            )
            for label, paths in train.items()
        }
        val_fold = {
            label: Reloader(
                CCTADataset(
                    paths,
                    [label] * len(paths),
                    val_patch_size,
                    max_HU_diff=max_HU_diff,
                    rng=rng,
                ),
                device_type,
                infinite=False,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers = num_workers[1]
            )
            for label, paths in val.items()
        }
        ret.append((train_fold, val_fold))

    return ret


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
    return {
        k: vars[k]
        for k in [
            "lr",
            "betas",
            "milestones",
            "lr_gamma",
            "HULoss_args",
            "max_HU_diff",
            "generator_args",
            "discriminator_args",
            "train_patch_size",
            "train_batch_size",
            "val_patch_size",
            "val_batch_size",
            "dataset_paths",
            "train_transform_args",
            "train_iterations",
            "train_generator_every",
            "seed",
            "fold_idx",
            "checkpoint_every",
            "validate_every",
            "log_every",
            "num_workers"
        ]
    } | {
        k: object_name(vars[k])
        for k in [
            "generator",
            "generator_optim",
            "generator_lr_scheduler",
            "discriminator",
            "discriminator_optim",
            "discriminator_lr_scheduler",
            "train_transform",
        ]
    }
