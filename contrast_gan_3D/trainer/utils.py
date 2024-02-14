from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.constants import DEFAULT_SEED, TRAIN_PATCH_SIZE
from contrast_gan_3D.data.CCTADataset import CCTADataset
from contrast_gan_3D.trainer.Reloader import Reloader


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
    *dataset_paths: Iterable[Union[str, Path]],
    train_transform: Optional[Callable[[dict], dict]] = None,
    seed: int = DEFAULT_SEED,
    n_folds: int = 5
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
                    rng=rng,
                    transform=train_transform,
                ),
                reload=True,
                batch_size=train_batch_size,
                shuffle=True,
            )
            for label, paths in train.items()
        }
        val_fold = {
            label: Reloader(
                CCTADataset(paths, [label] * len(paths), val_patch_size, rng=rng),
                reload=False,
                batch_size=val_batch_size,
                shuffle=False,
            )
            for label, paths in val.items()
        }
        ret.append((train_fold, val_fold))

    return ret
