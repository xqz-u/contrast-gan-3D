from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader
from torch.utils.data import default_collate

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.data.utils import minmax_norm
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
        normalize_range: Optional[Tuple[float, float]] = None,
        dataset_mean: Optional[float] = None,
        infinite: bool = True,
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
        self.dataset_mean = dataset_mean
        self.normalize_range = normalize_range

    def __len__(self) -> int:
        return len(self.indices)

    def scale(self, arr: np.ndarray) -> np.ndarray:
        if self.normalize_range is not None:  # scale to [0, 1]
            arr = minmax_norm(arr, self.normalize_range)
        if self.dataset_mean is not None:  # 0-center
            arr -= self.dataset_mean
        return arr

    def generate_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict, str]:
        with HD5Scan(self._data[idx]) as patient:
            ccta, arteries_mask = patient.ccta[::], patient.labelmap[::]  # HWD
            # pad if image is smaller than `patch_size`
            ccta = pad_nd_image(ccta, self.patch_size)
            # `crop` wants BCWHD
            patch, mask = crop(
                ccta.swapaxes(0, 1)[None, None],
                arteries_mask.swapaxes(0, 1)[None, None],
                self.patch_size,
                crop_type="random",
                # crop_type="center",
            )
            patch, mask = patch.swapaxes(2, 3), mask.swapaxes(2, 3)
        return self.scale(patch), mask, patient.meta, patient.name

    def generate_train_batch(self) -> dict:
        data = np.zeros(self.batch_shape, dtype=np.float32)  # BCHWD
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
