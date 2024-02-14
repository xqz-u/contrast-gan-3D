import torch
from torch.utils.data import DataLoader

from contrast_gan_3D.data.CCTADataset import CCTADataset


# adds channel dimension and coneverts centerline masks to boolean
def my_collate(batch: list) -> dict:
    ret = {
        k: torch.stack([torch.from_numpy(s.pop(k)) for s in batch])
        .unsqueeze(1)
        .to(dtype)
        for k, dtype in zip(["data", "seg"], [torch.float32, torch.bool])
    }
    return ret | torch.utils.data.default_collate(batch)


class Reloader:
    def __init__(self, dataset: CCTADataset, reload: bool = True, **dataloader_kwargs):
        self.dataset = dataset
        self.reload = reload
        dataloader_kwargs["collate_fn"] = dataloader_kwargs.get(
            "collate_fn", my_collate
        )
        self.dataloader = DataLoader(self.dataset, **dataloader_kwargs)
        self.dataloader_iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        try:
            batch = next(self.dataloader_iterator)
        except StopIteration:
            if self.reload:
                self.dataloader_iterator = iter(self.dataloader)
                batch = next(self.dataloader_iterator)
            else:
                raise
        return batch
