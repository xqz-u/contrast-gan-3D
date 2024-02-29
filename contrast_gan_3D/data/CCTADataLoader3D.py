from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader
from torch.utils.data import default_collate

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


# heavily inspired from
# https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
class CCTADataLoader3D(DataLoader):
    def __init__(
        self,
        data: List[Union[str, Path]],
        patch_size: Shape3D,
        batch_size: int,
        infinite: bool,
        max_HU_diff: Optional[int] = None,
        shuffle=True,
        num_threads_in_multithreaded=1,
        seed_for_shuffle: Optional[int] = None,
        return_incomplete=False,
        sampling_probabilities=None,
    ):
        super().__init__(
            data,
            batch_size,
            num_threads_in_multithreaded=num_threads_in_multithreaded,
            seed_for_shuffle=seed_for_shuffle,
            return_incomplete=return_incomplete,
            shuffle=shuffle,
            infinite=infinite,
            sampling_probabilities=sampling_probabilities,
        )
        self.patch_size = patch_size
        self.batch_shape = (self.batch_size, 1, *self.patch_size)
        self.indices = list(range(len(data)))
        self.max_HU_diff = max_HU_diff

    def __len__(self) -> int:
        return len(self.indices)

    def scale(self, arr: np.ndarray, patient: HD5Scan, ccta: np.ndarray) -> np.ndarray:
        if self.max_HU_diff is not None:
            # min is precomputed for each scan to avoid loading it entirely
            # into memory - only patches are needed
            if "min" in patient.meta:
                patient_min = patient.meta["min"]
            else:
                logger.info(
                    "Patch for '%s' - no precomputed scan-level min, computing it...",
                    patient.meta["path"],
                )
                patient_min = ccta.min()
            # the Tanh in the model forces its output to be in [1,-1], and the patch
            # intensity scaling below allows the Tanh output to be mapped to
            # `max_HU_diff`
            arr = (arr + patient_min) / self.max_HU_diff
        return arr

    def generate_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict, str]:
        with HD5Scan(self._data[idx], rng=self.rs) as patient:
            ccta, arteries_mask = patient.ccta[::], patient.labelmap[::]
            # pad if image is smaller than `patch_size`
            ccta = pad_nd_image(ccta, self.patch_size)
            # add BC dimensions
            patch, mask = crop(
                ccta[None, None],
                arteries_mask[None, None],
                self.patch_size,
                crop_type="random",
            )
        return self.scale(patch, patient, ccta), mask, patient.meta, patient.name

    def generate_train_batch(self) -> dict:
        data = np.zeros(self.batch_shape, dtype=np.float32)  # BCWHD
        masks = np.zeros(self.batch_shape, dtype=np.uint8)
        metadata, names = [], []

        for i, idx in enumerate(self.get_indices()):
            patch, mask, meta, name = self.generate_one(idx)
            data[i], masks[i] = patch, mask
            metadata.append(meta), names.append(name)

        return {
            "data": data,
            "seg": masks,
            "meta": default_collate(metadata),
            "name": default_collate(names),
        }
