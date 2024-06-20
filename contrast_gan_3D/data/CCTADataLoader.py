import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.data import utils as data_u
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.utils import geometry as geom


# inspired from
# https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
class CCTADataLoader(DataLoader):
    def __init__(
        self,
        data: list[str],
        patch_shape: Shape3D,
        batch_size: int,
        rng: np.random.Generator,
        scaler: Scaler = lambda x: x,
        infinite: bool = True,
        shuffle=True,
        num_threads_in_multithreaded=1,
        seed_for_shuffle: int | None = None,
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

    def get_samplable_2D(
        self, data_and_seg: np.memmap | np.ndarray, meta: dict
    ) -> tuple[np.ndarray, bool]:
        sample_along_centerlines = self.rng.random() < 0.5
        if sample_along_centerlines:
            # extract patch from slice with *at least one* ensured centerline
            centerlines = meta["centerlines_world"]
            centerline_idx = self.rng.integers(0, len(centerlines))
            x, y, z = geom.world_to_image_coords(
                centerlines[centerline_idx, :3], meta["offset"], meta["spacing"]
            )
            bbox = geom.get_patch_bounds(
                self.patch_shape, data_and_seg[..., z, 0].shape, np.array([y, x])
            )
            indexer = [slice(*bbox[0]), slice(*bbox[1]), z]  # indexing on xyz
        else:  # extract patch from random slice
            indexer = [..., self.rng.choice(data_and_seg.shape[2])]  # indexing on xyz
        # patch and mask are in WH order, final batch shape: BCWH
        return (data_and_seg[*indexer, :], not sample_along_centerlines)

    def get_samplable_3D(
        self, data_and_seg: np.memmap | np.ndarray, *_
    ) -> tuple[np.ndarray, bool]:
        return data_and_seg, True  # WHD

    def generate_one(self, patient_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        ccta_and_seg, meta = data_u.load_patient(patient_path)  # 4D: WHD[HU,label]
        ccta_and_seg, do_crop = self.get_samplable(ccta_and_seg, meta)
        ccta_and_seg = ccta_and_seg[None, None]  # `crop` wants BCWH(D)
        patch, mask = ccta_and_seg[..., 0], ccta_and_seg[..., 1]
        if do_crop:
            # pad if image is smaller than `patch_size`
            ccta_and_seg = pad_nd_image(ccta_and_seg, (*self.patch_shape, 2))
            # convert to float to avoid rounding errors while cropping
            ccta_and_seg = ccta_and_seg.astype(np.float32)
            patch, mask = crop(
                ccta_and_seg[..., 0],
                ccta_and_seg[..., 1],
                self.patch_shape,
                crop_type="random",
            )
        return self.scaler(patch), mask, meta["name"]

    # NOTE could try #4 from
    # https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869
    # to reduce CPU to GPU copy size
    def generate_train_batch(self) -> dict:
        data = np.zeros(self.batch_shape, dtype=np.float32)  # BCWH(D)
        masks = np.zeros(self.batch_shape, dtype=np.float32)
        names, paths = [], []

        for i, idx in enumerate(self.get_indices()):
            patient_path = self._data[idx]
            patch, mask, name = self.generate_one(patient_path)
            data[i], masks[i] = patch, mask
            names.append(name), paths.append(patient_path)

        return {"data": data, "seg": masks, "name": names, "path": paths}
