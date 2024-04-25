from typing import Tuple

import numpy as np
from patchly.sampler import GridSampler
from torch import Tensor
from torch.utils.data import Dataset

from contrast_gan_3D.data.Scaler import Scaler


class CCTAEvalDataset3D(Dataset):
    def __init__(self, scaler: Scaler, sampler: GridSampler):
        super().__init__()
        self.sampler = sampler
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        patch, bbox = self.sampler[index]
        # add channel dimension and scale patch
        return self.scaler(patch)[None], bbox


class CCTAEvalDataset2D(Dataset):
    def __init__(self, scaler: Scaler, ccta: np.ndarray):
        super().__init__()
        self.scaler = scaler
        self.ccta = ccta.astype(np.float32)

    def __len__(self) -> int:
        return self.ccta.shape[-1]

    def __getitem__(self, index: int) -> Tensor:
        # add channel dimension and scale patch, shape: (1,512,512)
        return self.scaler(self.ccta[..., index])[None]
