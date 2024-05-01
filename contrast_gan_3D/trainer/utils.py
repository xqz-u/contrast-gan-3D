import importlib
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from sklearn.model_selection import StratifiedKFold, train_test_split

from contrast_gan_3D.alias import BGenAugmenter, FoldType, Shape3D
from contrast_gan_3D.constants import DEFAULT_SEED
from contrast_gan_3D.data.CCTADataLoader import CCTADataLoader
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.utils import object_name


def find_latest_checkpoint(ckpt_dir: Union[Path, str]) -> Optional[Path]:
    ckpt_dir, cont = Path(ckpt_dir), []
    for file_path in ckpt_dir.glob("*.pt"):
        try:
            ckpt_number = int(file_path.stem)  # expects paths 0.pt, 1.pt, ...
            cont.append(ckpt_number)
        except ValueError:
            ...
    return None if not len(cont) else ckpt_dir / f"{max(cont)}.pt"


def cval_paths(
    n_folds: int,
    *dataset_paths: Iterable[Union[Path, str]],
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[List[FoldType], List[FoldType]]:
    X, Y = [], []
    for df_path in dataset_paths:
        df = pd.read_excel(df_path)
        X += df["path"].values.tolist()
        Y += df["label"].values.tolist()
    X, Y = np.array(X), np.array(Y)

    if n_folds == 1:
        xtrain, xval, ytrain, yval = train_test_split(
            X, Y, test_size=test_size, shuffle=True, stratify=Y, random_state=seed
        )
        train, val = [list(zip(xtrain, ytrain))], [list(zip(xval, yval))]
    else:
        train, val = [], []
        cval = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, val_idx in cval.split(X, Y):
            train.append(list(zip(X[train_idx], Y[train_idx])))
            val.append(list(zip(X[val_idx], Y[val_idx])))

    return train, val


def divide_scans_in_fold(fold: FoldType) -> Dict[int, List[Union[str, Path]]]:
    ret = defaultdict(list)
    for path, label in fold:
        ret[label].append(path)
    return ret


def create_dataloaders(
    train_fold: FoldType,
    val_fold: FoldType,
    train_patch_size: Shape3D,
    val_patch_size: Shape3D,
    train_batch_sizes: Dict[int, int],
    val_batch_sizes: Dict[int, int],
    rng: np.random.Generator,
    scaler: Type[Scaler] = lambda x: x,
    num_workers: Tuple[int, int] = (1, 1),
    train_transform: Optional[Callable[[dict], dict]] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[int, BGenAugmenter], Dict[int, BGenAugmenter]]:
    pin_memory = torch.cuda.is_available()

    train_by_lab = divide_scans_in_fold(train_fold)
    train_loaders = {
        label: NonDetMultiThreadedAugmenter(
            CCTADataLoader(
                paths,
                train_patch_size,
                train_batch_sizes[label],
                rng,
                scaler=scaler,
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
            CCTADataLoader(
                paths,
                val_patch_size,
                val_batch_sizes[label],
                rng,
                scaler=scaler,
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


def config_from_globals(vars: dict) -> dict:
    return (
        {
            k: vars[k]
            for k in [
                "lr",
                "betas",
                "milestones",
                "lr_gamma",
                "weight_clip",
                "max_HU_delta",
                "desired_HU_bounds",
                "HU_norm_range",
                "generator_args",
                "critic_args",
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
                "n_cval_folds",
                "checkpoint_every",
                "validate_every",
                "log_every",
                "log_images_every",
                "num_workers",
            ]
        }
        | {
            k: object_name(o if not isinstance((o := vars[k]), partial) else o.func)
            for k in [
                "generator_optim_class",
                "generator_lr_scheduler_class",
                "critic_optim_class",
                "critic_lr_scheduler_class",
            ]
        }
        | {k: str(vars[k]) for k in ["train_transform", "scaler", "logger_interface"]}
    )
