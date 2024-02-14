from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from contrast_gan_3D.alias import Shape3D
from contrast_gan_3D.data.HD5Scan import HD5Scan


class CCTADataset(Dataset):
    def __init__(
        self,
        hd5_paths: List[Union[Path, str]],
        labels: List[int],
        patch_size: Union[Shape3D, int],
        max_HU_diff: Optional[int] = None,
        transform: Optional[Callable[[dict], dict]] = None,
        rng: Optional[Union[np.random.RandomState, np.random.Generator]] = None,
    ):
        self.hd5_paths = hd5_paths
        self.labels = labels
        self.patch_size = patch_size
        if isinstance(self.patch_size, int) and self.patch_size == -1:
            self.patch_size = self._calculate_patch_size()
        self.rng = rng or np.random.default_rng()
        self.transform = transform
        self.max_HU_diff = max_HU_diff

    # NOTE padding needed if using "max" strategy
    def _calculate_patch_size(self, mode: str = "min") -> Shape3D:
        shapes = []
        for datapoint in self.hd5_paths:
            with HD5Scan(datapoint) as patient_scan:
                shapes.append(patient_scan.ccta.shape)
        shapes = np.array(shapes)
        return (shapes.min if mode == "min" else shapes.max)(0)

    def scale_img(self, img: torch.Tensor) -> torch.Tensor:
        # the Tanh in the model forces its output to be in [1,-1], and the patch
        # intensity scaling below allows the Tanh output to be mapped to
        # `max_HU_diff`
        return (img + img.min()) / self.max_HU_diff

    def __len__(self) -> int:
        return len(self.hd5_paths)

    def __getitem__(self, index: int) -> dict:
        ret = {"label": self.labels[index]}
        with HD5Scan(self.hd5_paths[index], self.rng) as patient:
            patch_origin, patch, patch_mask = patient.sample_volume(self.patch_size)
            ret["data"] = patch
            ret["seg"] = patch_mask
            ret["meta"] = patient.meta
            ret["names"] = patient.name
            ret["patch_origin"] = patch_origin

        if self.max_HU_diff is not None:
            patch = self.scale_img(patch)

        if self.transform is not None:
            # NOTE assumes a transform operating on a (b,c,x,y,z) tensor, like
            # batchgenerator's ones
            transformed = self.transform(
                data=ret["data"][None, None], seg=ret["seg"][None, None]
            )
            for k, v in transformed.items():
                ret.update({k: v.squeeze()})

        return ret
