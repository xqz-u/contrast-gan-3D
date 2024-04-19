from pathlib import Path
from typing import Optional, Union

import h5py

from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils


class HD5Scan:
    def __init__(self, h5_path: Union[str, Path]):
        self.path = h5_path
        self.name = io_utils.stem(self.path)
        # attributes set within __enter__
        self.hd5_file = None
        self.meta = {}
        self.labelmap: Optional[h5py.Dataset] = None
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
            return self.hd5_file["ccta"]["ccta"]  # HWD

    @ccta.setter
    def ccta(self, _):
        ...

    @property
    def centerlines(self) -> Optional[h5py.Dataset]:
        if self.hd5_file is not None:
            return self.hd5_file["ccta"]["centerlines"]  # HWD

    @centerlines.setter
    def centerlines(self, _):
        ...

    def _parse_centerlines(self):
        self._assert_loaded()
        self.labelmap = self.hd5_file["ccta"].get("centerlines_seg")
        # safety net, but segmentations should generally be in the .h5 files
        if self.labelmap is None:
            self.labelmap = geom.world_to_grid_coords(
                self.centerlines[..., 3],
                self.meta["offset"],
                self.meta["spacing"],
                self.ccta.shape,
            )

    def __repr__(self) -> str:
        return f"<hd5: {self.hd5_file} path: {str(self.path)!r} ccta: {self.ccta}>"
