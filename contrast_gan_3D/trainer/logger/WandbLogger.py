from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid
from wandb.sdk.wandb_run import Run

import wandb
from contrast_gan_3D.alias import Array
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.data.utils import minmax_norm
from contrast_gan_3D.utils import geometry as geom


@dataclass
class WandbLogger:
    scaler: Type[Scaler]
    sample_size: int = 64
    run: Run = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    cmap: colors.Colormap = field(default_factory=lambda: cm.RdBu)
    figsize: Tuple[int, int] = (10, 10)
    grid_args: dict = field(
        default_factory=lambda: {"normalize": True, "value_range": (VMIN, VMAX)}
    )

    def setup_wandb_run(self, run: Run):
        # https://community.wandb.ai/t/log-stats-with-different-global-steps/4375/3
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        self.run = run

    def log_loss(self, loss_dict: Dict[str, Tensor], iteration: int, stage: str):
        self.run.log(
            {f"{stage}/{k}": v for k, v in loss_dict.items()} | {"step": iteration}
        )

    def reconstruct_sample(self, slices: Array) -> Array:
        return self.scaler.unscale(slices)

    def reconstruct_optimized_sample(self, slices: Array) -> Array:
        return self.scaler.unscale(slices)

    def create_indexer(self, batch_shape: Tuple[int, ...]) -> list:
        sample_idx = self.rng.integers(batch_shape[0])
        slice_idxs = sorted(
            self.rng.choice(batch_shape[-1], size=self.sample_size, replace=False)
        )
        return [sample_idx, ..., slice_idxs]

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
    ) -> list:
        indexer = self.create_indexer(scans.shape)
        sample_idx = indexer[0]
        fig, caption = None, names[sample_idx]
        workspace, caption_cp = f"{stage}/images/{scan_type}", caption
        buffer = []

        # show centerlines by scattering manually during training
        if stage == "train" and masks is not None:
            ctls = masks[indexer].permute(3, 0, 1, 2).to(torch.float16)
            ctls_grid = make_grid(ctls)
            # DHW -> HWD (yxz)
            cart = geom.grid_to_cartesian_coords(ctls_grid.permute(1, 2, 0))
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(
                cart[:, 1],
                cart[:, 0],
                c="red",
                s=plt.rcParams["lines.markersize"] * 0.8,
            )
            caption_cp = f"{caption} {np.prod(cart.shape)}/{np.prod(masks[sample_idx].shape)} centerlines"

        slices = self.reconstruct_sample(scans[indexer])
        fig = WandbLogger.create_grid_figure(slices, fig, **self.grid_args)
        buffer.append(((f"{workspace}/sample", it, fig), {"caption": caption_cp}))

        if reconstructions is not None:
            recon = self.reconstruct_optimized_sample(reconstructions[indexer])
            fig = WandbLogger.create_grid_figure(recon, **self.grid_args)
            buffer.append(
                ((f"{workspace}/reconstruction", it, fig), {"caption": caption})
            )

        if attenuations is not None:
            fig = self.create_attenuation_grid(attenuations, indexer)
            buffer.append(((f"{workspace}/attenuation", it, fig), {"caption": caption}))

        return buffer

    def create_attenuation_grid(self, attenuations: Tensor, indexer: list) -> Figure:
        # normalize [-1, 1]->[0, 1] for colormap (min and max from entire sample)
        attn_sample = attenuations[indexer[0]]
        low, high = attn_sample.min().item(), attn_sample.max().item()
        attn = minmax_norm(attenuations[indexer], (low, high))
        attn = self.cmap(attn).squeeze().transpose(3, 0, 1, 2)
        # add colorbar
        fig, ax = plt.subplots(figsize=self.figsize)
        norm = colors.Normalize(low, high)
        mappable = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
        cbar.set_ticks(np.linspace(low, high, 5))
        return WandbLogger.create_grid_figure(attn, fig, tight=False)

    @staticmethod
    def create_grid_figure(
        slices: Union[Tensor, np.ndarray],
        fig: Optional[Figure] = None,
        tight: bool = True,
        **grid_args,
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
        if tight:
            fig.tight_layout()
        return fig

    def log_wandb_image(
        self,
        tag: str,
        it: int,
        fig: Figure,
        caption: Optional[str] = None,
    ):
        image = wandb.Image(fig, caption=caption)
        # image.image.show()
        self.run.log({tag: image, "step": it})
        plt.close(fig)

    def log_images(self, buffer: List[tuple]):
        for args, kwargs in buffer:
            self.log_wandb_image(*args, **kwargs)
