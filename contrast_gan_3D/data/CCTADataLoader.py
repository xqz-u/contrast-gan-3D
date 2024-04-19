from typing import List, Optional, Tuple

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader
from torch.utils.data import default_collate

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


# inspired from
# https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
class CCTADataLoader(DataLoader):
    def __init__(
        self,
        data: List[str],
        patch_shape: Shape3D,
        batch_size: int,
        rng: np.random.Generator,
        scaler: Scaler = lambda x: x,
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
        self.patch_shape = np.array(patch_shape)
        self.batch_shape = (self.batch_size, 1, *self.patch_shape)
        self.indices = list(range(len(data)))
        self.scaler = scaler
        self.rng = rng
        self.get_samplable = self.get_samplable_3D
        if len(patch_shape) < 3:
            self.get_samplable = self.get_samplable_2D

    def __len__(self) -> int:
        return len(self.indices)

    def get_samplable_2D(self, patient: HD5Scan) -> Tuple[np.ndarray, np.ndarray, bool]:
        sample_along_centerlines = self.rng.random() < 0.5
        if sample_along_centerlines:
            # extract patch from slice with *at least one* ensured centerline
            centerline_idx = self.rng.integers(0, len(patient.centerlines))
            x, y, z = geom.world_to_image_coords(
                patient.centerlines[centerline_idx, :3],
                patient.meta["offset"],
                patient.meta["spacing"],
            )
            bbox = geom.get_patch_bounds(
                self.patch_shape, patient.ccta[..., z].shape, np.array([y, x])
            )
            indexer = [slice(*bbox[0]), slice(*bbox[1]), z]
        else:  # extract patch from random slice
            indexer = [..., self.rng.choice(patient.ccta.shape[-1])]
        # patch and mask are in HW order
        return (
            patient.ccta[*indexer],
            patient.labelmap[*indexer],
            not sample_along_centerlines,
        )

    # NOTE would be better to use a memmap recognized by batchegenerators
    def get_samplable_3D(self, patient: HD5Scan) -> Tuple[np.ndarray, np.ndarray, bool]:
        return patient.ccta[::], patient.labelmap[::], True  # HWD

    def generate_one(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict, str]:
        with HD5Scan(self._data[idx]) as patient:
            ccta, arteries_mask, do_crop = self.get_samplable(patient)
        patch = ccta.swapaxes(0, 1)[None, None]
        mask = arteries_mask.swapaxes(0, 1)[None, None]
        if do_crop:
            # pad if image is smaller than `patch_size`
            patch = pad_nd_image(patch, self.patch_shape)
            # `crop` wants BCWH(D)
            patch, mask = crop(patch, mask, self.patch_shape, crop_type="random")
        patch, mask = patch.swapaxes(2, 3), mask.swapaxes(2, 3)
        return self.scaler(patch), mask, patient.meta, patient.name

    # NOTE could try #4 from
    # https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869
    # to reduce CPU to GPU copy size
    def generate_train_batch(self) -> dict:
        data = np.zeros(self.batch_shape, dtype=np.float32)  # BCHW(D)
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
