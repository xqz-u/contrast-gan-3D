from typing import Tuple

from patchly.sampler import GridSampler
from torch import Tensor
from torch.utils.data import Dataset

from contrast_gan_3D.data.Scaler import Scaler


class CCTAEvalDataset(Dataset):
    def __init__(self, sampler: GridSampler, scaler: Scaler):
        super().__init__()
        self.sampler = sampler
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        patch, bbox = self.sampler[index]
        # add channel dimension and scale patch
        return self.scaler(patch)[None], bbox
