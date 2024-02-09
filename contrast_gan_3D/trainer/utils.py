from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from contrast_gan_3D.config import DEFAULT_SEED


def crossval_paths(
    *dataset_paths: Iterable[Union[Path, str]], seed: int = DEFAULT_SEED
) -> Tuple[List[List[Tuple[Union[str, Path], int]]], ...]:
    X, Y, train, val = [], [], [], []

    for df_path in dataset_paths:
        df = pd.read_excel(df_path)
        X += df["path"].values.tolist()
        Y += df["label"].values.tolist()
    X, Y = np.array(X), np.array(Y)

    cval = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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


# adds channel dimension and coneverts centerline masks to boolean
def my_collate(batch: list) -> dict:
    ret = {
        k: torch.stack([torch.from_numpy(s.pop(k)) for s in batch])
        .unsqueeze(1)
        .to(dtype)
        for k, dtype in zip(["data", "seg"], [torch.float32, torch.bool])
    }
    return ret | torch.utils.data.default_collate(batch)
