from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.config import DEFAULT_SEED
from contrast_gan_3D.constants import MODEL_PATCH_SIZE
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils
from contrast_gan_3D.utils import visualization as viz


# TODO make `labelmap` part of the original HD5 file to avoid computing it
# each time a patient's file is opened
# TODO better API for `sample_volume`, should return only what's needed - patch
# and small mask. The ones used for visualization should be in another method
class HD5Dataset:
    def __init__(
        self, h5_path: Union[str, Path], rng: Optional[np.random.Generator] = None
    ):
        self.hd5_file = None
        self.meta = {}

        self.labelmap = None
        self.path = h5_path
        self.name = io_utils.stem(self.path)
        self.rng = rng or np.random.default_rng(seed=DEFAULT_SEED)

    def __enter__(self) -> "HD5Dataset":
        self.ccta, self.meta, self.hd5_file = io_utils.load_h5_image(self.path)
        self._parse_centerlines()
        return self

    def __exit__(self, *_):
        self.hd5_file.close()
        self.hd5_file = None

    def _assert_legal(self):
        assert self.ccta is not None, f"HD5 file {str(self.path)!r} is closed"

    @property
    def ccta(self) -> Optional[h5py.Dataset]:
        if self.hd5_file is not None:
            return self.hd5_file["ccta"]["ccta"]

    @ccta.setter
    def ccta(self, _):
        ...

    @property
    def centerlines(self) -> Optional[h5py.Dataset]:
        if self.hd5_file is not None:
            return self.hd5_file["ccta"]["centerlines"]

    @centerlines.setter
    def centerlines(self, _):
        ...

    def _parse_centerlines(self):
        self._assert_legal()
        self.centerlines_img_coords = geom.world_to_image_coords(
            self.meta["centerlines"][..., :3], self.meta["offset"], self.meta["spacing"]
        )
        self.labelmap = torch.zeros(self.ccta.shape, dtype=torch.int8)
        self.labelmap[*[self.centerlines_img_coords[:, i] for i in range(3)]] = 1

    def sample_volume(
        self, patch_size: Union[np.ndarray, Shape3D] = MODEL_PATCH_SIZE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_legal()
        # extract random patch centered at `origin`
        patch, _, origin = geom.extract_random_3D_patch(
            self.ccta, patch_size, rng=self.rng
        )
        # extract a binary mask with the same size as `patch` with 1s on the
        # centerlines contained in the patch, and the same but with size
        # `self.labelmap.shape`
        binary_patch, whole_image_binary_patch = geom.extract_3D_patch(
            self.labelmap, patch_size, origin
        )
        return (
            torch.from_numpy(origin),
            torch.from_numpy(patch),
            binary_patch,
            whole_image_binary_patch,
        )

    def __repr__(self) -> str:
        return f"<hd5: {self.hd5_file} path: {str(self.path)!r} ccta: {self.ccta}>"
