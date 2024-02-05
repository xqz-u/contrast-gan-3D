from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.config import DEFAULT_SEED
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils


# NOTE `sample_volume` might be a bottleneck with `batchgenerator`'s dataloaders,
# consider switching to their patch cropping implementation
class HD5Scan:
    def __init__(
        self,
        h5_path: Union[str, Path],
        rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
    ):
        self.path = h5_path
        self.rng = rng or np.random.default_rng(seed=DEFAULT_SEED)
        self.name = io_utils.stem(self.path)
        # attributes set within __enter__
        self.hd5_file = None
        self.meta = {}
        self.labelmap = None
        self.centerlines_img_coords = None

    def __enter__(self) -> "HD5Scan":
        self.ccta, self.meta, self.hd5_file = io_utils.load_h5_image(self.path)
        if "path" not in self.meta:
            self.meta["path"] = self.path
        self._parse_centerlines()
        return self

    def __exit__(self, *_):
        self.hd5_file.close()
        self.hd5_file = None

    def _assert_loaded(self):
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
        self._assert_loaded()
        self.labelmap = self.hd5_file["ccta"].get("centerlines_seg")
        # safety net, but segmentations should generally be in the .h5 files
        if self.labelmap is None:
            self.labelmap = geom.world_to_grid_coords(
                self.centerlines,
                self.meta["offset"],
                self.meta["spacing"],
                self.ccta.shape,
            )

    def sample_volume(
        self, patch_size: Union[np.ndarray, Shape3D]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._assert_loaded()
        # random patch centered at `origin`
        patch, origin = geom.extract_random_3D_patch(
            self.ccta, patch_size, rng=self.rng
        )
        # extract a binary mask with the same size as `patch` with 1s on the
        # centerlines contained in the patch, and the same but with size
        # `self.labelmap.shape`
        binary_patch = geom.extract_3D_patch(self.labelmap, patch_size, origin)
        return (origin, patch, binary_patch)

    def __repr__(self) -> str:
        return f"<hd5: {self.hd5_file} path: {str(self.path)!r} ccta: {self.ccta}>"
