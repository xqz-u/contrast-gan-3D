from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from batchgenerators.dataloading.data_loader import DataLoader
from sklearn.model_selection import StratifiedKFold

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.config import DEFAULT_SEED
from contrast_gan_3D.constants import MODEL_PATCH_SIZE
from contrast_gan_3D.data.HD5Scan import HD5Scan


# heavily inspired from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
class CCTADataLoader(DataLoader):
    def __init__(
        self,
        data: List[Tuple[Union[str, Path], int]],
        batch_size: int,
        patch_size: Shape3D = MODEL_PATCH_SIZE,
        num_threads_in_multithreaded=1,
        seed_for_shuffle: Optional[int] = None,
        return_incomplete=False,
        shuffle=True,
        infinite: bool = True,
        sampling_probabilities=None,
    ):
        super().__init__(
            data,
            batch_size,
            num_threads_in_multithreaded,
            seed_for_shuffle,
            return_incomplete,
            shuffle,
            infinite,
            sampling_probabilities,
        )
        self.patch_size = patch_size
        self.batch_shape = (self.batch_size, 1, *self.patch_size)
        self.indices = list(range(len(data)))

    # NOTE the returned `HD5Scan` is closed and should be opened inside a `with`
    # statement
    def load_patient(
        self, patient_path: Union[str, Path], label: int
    ) -> Tuple[HD5Scan, int]:
        return HD5Scan(patient_path, rng=self.rs), label

    def generate_train_batch(self) -> dict:
        data = np.zeros(self.batch_shape, dtype=np.float32)
        # NOTE is this the right dtype? binary mask interacting with float32 later
        masks = np.zeros(self.batch_shape, dtype=np.uint8)
        origins = np.zeros((self.batch_size, 3), dtype=np.int16)
        metadata, patient_names = [], []

        patients_scans, labels = list(
            zip(*[self.load_patient(*self._data[i]) for i in self.get_indices()])
        )

        for i, patient in enumerate(patients_scans):
            with patient:
                (patch_origin, patch, patch_mask) = patient.sample_volume(
                    self.patch_size
                )
            origins[i] = patch_origin
            data[i] = patch
            masks[i] = patch_mask

            metadata.append(patient.meta)
            patient_names.append(patient.name)

        return {
            "data": data,
            "seg": masks,
            "labels": labels,
            "meta": metadata,
            "names": patient_names,
            "patch_origin": origins,
        }

    # NOTE probably need diff train/val batch & patch sizes
    @classmethod
    def create_from_cval(
        cls: "CCTADataLoader",
        batch_size: int,
        *dataset_paths: Iterable[Union[Path, str]],
        patch_size: Optional[Shape3D] = MODEL_PATCH_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> Dict[str, List["CCTADataLoader"]]:
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

        make = lambda x: [
            cls(xx, batch_size=batch_size, patch_size=patch_size) for xx in x
        ]
        return {"train": make(train), "val": make(val)}
