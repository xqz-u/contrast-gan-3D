import torch
from torch.utils.data import DataLoader, Dataset


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
    def __init__(
        self,
        dataset: Dataset,
        device_type: str,
        infinite: bool = True,
        **dataloader_kwargs
    ):
        self.dataset = dataset
        self.infinite = infinite
        dataloader_kwargs["collate_fn"] = dataloader_kwargs.get(
            "collate_fn", my_collate
        )
        self.dataloader = DataLoader(
            self.dataset, pin_memory=device_type == "cuda", **dataloader_kwargs
        )
        self.reset()

    def reset(self):
        self.dataloader_iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        try:
            batch = next(self.dataloader_iterator)
        except StopIteration:
            self.reset()
            if self.infinite:
                batch = next(self.dataloader_iterator)
            else:
                # raising allows to iterate over the whole dataloader, and
                # still to reset it once it is exhausted (e.g. intermittent
                # validation)
                raise
        return batch
