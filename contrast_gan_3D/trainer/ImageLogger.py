from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid

import wandb
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.data.utils import minmax_norm
from contrast_gan_3D.utils import geometry as geom


@dataclass
class ImageLogger:
    norm_denominator: int
    shift: int
    sample_size: int = 64
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    cmap: colors.Colormap = field(default_factory=lambda: cm.RdBu)
    figsize: Tuple[int, int] = (10, 10)

    def create_indexer(self, batch_shape: Tuple[int, ...]) -> list:
        sample_idx = self.rng.integers(batch_shape[0])
        slice_idxs = sorted(
            self.rng.choice(batch_shape[-1], size=self.sample_size, replace=False)
        )
        return [sample_idx, ..., slice_idxs]

    # log 64 slices of a random sample in a batch
    def __call__(
        self,
        scans: Tensor,
        it: int,
        stage: str,
        scan_type: str,
        names: List[str],
        masks: Optional[Tensor] = None,
        reconstructions: Optional[Tensor] = None,
        attenuations: Optional[Tensor] = None,
    ):
        indexer = self.create_indexer(scans.shape)
        sample_idx = indexer[0]
        grid_args = {"normalize": True, "value_range": (VMIN, VMAX)}
        fig, caption = None, names[sample_idx]
        workspace, caption_cp = f"{stage}/images/{scan_type}", caption

        # show centerlines by scattering manually during training
        if stage == "train":
            ctls = masks[indexer].permute(3, 0, 1, 2).to(torch.float16)
            ctls_grid = make_grid(ctls)
            # DHW -> HWD (yxz)
            cart = geom.grid_to_cartesian_coords(ctls_grid.cpu().permute(1, 2, 0))
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(
                cart[:, 1],
                cart[:, 0],
                c="red",
                s=plt.rcParams["lines.markersize"] * 0.8,
            )
            caption_cp = f"{caption} {np.prod(cart.shape)}/{np.prod(masks[sample_idx].shape)} centerlines"

        slices = scans[indexer] * self.norm_denominator + self.shift
        fig = ImageLogger.create_grid_figure(slices.cpu(), fig, **grid_args)
        self.log_wandb_image(f"{workspace}/sample", it, fig, caption=caption_cp)

        if reconstructions is not None:
            recon = reconstructions[indexer] * self.norm_denominator
            fig = ImageLogger.create_grid_figure(recon.cpu(), **grid_args)
            self.log_wandb_image(
                f"{workspace}/reconstruction", it, fig, caption=caption
            )

        if attenuations is not None:
            fig = self.create_attenuation_grid(attenuations, indexer)
            self.log_wandb_image(f"{workspace}/attenuation", it, fig, caption=caption)

    def create_attenuation_grid(self, attenuations: Tensor, indexer: list):
        # normalize [-1, 1]->[0, 1] for colormap (min and max from entire sample)
        attn_sample = attenuations[indexer[0]]
        low, high = attn_sample.min().item(), attn_sample.max().item()
        attn = minmax_norm(attenuations[indexer], (low, high))
        # https://discuss.pytorch.org/t/torch-utils-make-grid-with-cmaps/107471/2
        attn = np.apply_along_axis(self.cmap, 0, attn.detach().cpu().numpy()).squeeze()
        # add colorbar
        fig, ax = plt.subplots(figsize=self.figsize)
        norm = colors.Normalize(low, high)
        mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
        cbar.set_ticks(np.linspace(low, high, 5))
        return ImageLogger.create_grid_figure(attn, fig)

    @staticmethod
    def create_grid_figure(
        slices: Union[Tensor, np.ndarray], fig: Optional[Figure] = None, **grid_args
    ) -> Figure:
        if isinstance(slices, np.ndarray):
            slices = torch.from_numpy(slices)
        # CHWD -> DCHW -> C,HxD,WxD
        grid = make_grid(slices.permute(3, 0, 1, 2), **grid_args)
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax, *_ = fig.get_axes()
        ax.imshow(grid.permute(1, 2, 0))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.tight_layout()
        return fig

    def log_wandb_image(
        self,
        tag: str,
        it: int,
        fig: Figure,
        caption: Optional[str] = None,
    ):
        wandb.log({tag: wandb.Image(fig, caption=caption)}, step=it)
        plt.close(fig)
