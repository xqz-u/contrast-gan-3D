from dataclasses import dataclass, field
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid
from wandb.sdk.wandb_run import Run

import wandb
from contrast_gan_3D.alias import ArrayShape
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.data.utils import minmax_norm
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import swap_last_dim
from contrast_gan_3D.utils import visualization as viz


@dataclass
class WandbLogger:
    scaler: Type[Scaler]
    run: Optional[Run] = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    cmap: colors.Colormap = field(default_factory=lambda: cm.RdBu)
    figsize: Tuple[int, int] = (20, 20)
    use_caption: bool = field(init=False, default=True)
    norm_constants: dict = field(
        default_factory=lambda: {"normalize": True, "value_range": (VMIN, VMAX)},
        init=False,
    )

    def __post_init__(self):
        if self.run is not None:
            self.setup_wandb_run(self.run)

    def setup_wandb_run(self, run: Run):
        # https://community.wandb.ai/t/log-stats-with-different-global-steps/4375/3
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        self.run = run

    def log_loss(self, loss_dict: Dict[str, Tensor], iteration: int, stage: str):
        log_dict = {f"{stage}/{k}": v for k, v in loss_dict.items()}
        log_dict |= {"step": iteration}
        if self.run is not None:
            self.run.log(log_dict)
        else:
            pprint(log_dict)

    @staticmethod
    def create_indexer(
        batch_shape: ArrayShape, sample_size: int, rng: np.random.Generator
    ) -> list:
        # dim 0: B, dim -1: D
        sample_idx = rng.integers(batch_shape[0])
        slice_idxs = sorted(
            rng.choice(batch_shape[-1], size=sample_size, replace=False)
        )
        return [sample_idx, ..., slice_idxs]

    def pre_call_hook(self, *tensors: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
        return tensors

    def __call__(
        self,
        scans: Tensor,
        it: int,
        stage: str,
        scan_type: str,
        sample_size: int,
        names: List[str],
        masks: Optional[Tensor] = None,
        reconstructions: Optional[Tensor] = None,
        attenuations: Optional[Tensor] = None,
    ) -> list:
        scans, masks, reconstructions, attenuations = self.pre_call_hook(
            scans, masks, reconstructions, attenuations
        )

        indexer = self.create_indexer(scans.shape, sample_size, self.rng)
        sample_idx = indexer[0]
        fig, caption = None, names[sample_idx]
        workspace, caption_cp = f"{stage}/images/{scan_type}", caption
        buffer = []

        # show centerlines by scattering manually during training
        if stage == "train" and masks is not None:
            ctls = masks[indexer].permute(3, 0, 2, 1).to(torch.float16)  # CWHD -> DCHW
            ctls_grid = make_grid(ctls)
            cart = geom.grid_to_cartesian_coords(ctls_grid)  # DHW (zyx)
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.scatter(
                cart[:, 2],
                cart[:, 1],
                c="red",
                s=plt.rcParams["lines.markersize"] * 0.8,
            )
            caption_cp = f"{caption} {np.prod(cart.shape)}/{np.prod(masks[sample_idx].shape)} centerlines"

        cbar_low = self.scaler.unscale(scans[sample_idx].min())
        cbar_high = self.scaler.unscale(scans[sample_idx].max())
        fig = viz.plot_axial_slices(
            self.scaler.unscale(scans[indexer]),
            fig=fig,
            figsize=self.figsize,
            cbar_range=(cbar_low, cbar_high),
            **self.norm_constants,
        )
        buffer.append(((f"{workspace}/sample", it, fig), {"caption": caption_cp}))

        if reconstructions is not None:
            cbar_low = self.scaler.unscale(reconstructions[sample_idx].min())
            cbar_high = self.scaler.unscale(reconstructions[sample_idx].max())
            fig = viz.plot_axial_slices(
                self.scaler.unscale(reconstructions[indexer]),
                figsize=self.figsize,
                cbar_range=(cbar_low, cbar_high),
                **self.norm_constants,
            )
            buffer.append(
                ((f"{workspace}/reconstruction", it, fig), {"caption": caption})
            )

        if attenuations is not None:
            fig = self.create_attenuation_grid(attenuations, indexer)
            buffer.append(((f"{workspace}/attenuation", it, fig), {"caption": caption}))
        return buffer

    def create_attenuation_grid(
        self, attenuations: Tensor, indexer: list, scale_by_factor: bool = True
    ) -> Figure:
        # normalize to [0, 1] for colormap (min and max from entire scan)
        attn_sample = attenuations[indexer[0]]
        low, high = attn_sample.min(), attn_sample.max()
        attn = minmax_norm(attenuations[indexer], (low, high))
        # `cmap` adds new channel dimension, make it the first one
        attn = swap_last_dim(self.cmap(attn).squeeze())
        if hasattr(self.scaler, "factor") and scale_by_factor:
            # map generator output [-1,1] to used HU limits
            factor = self.scaler.factor
            low, high = low * factor, high * factor
        return viz.plot_axial_slices(
            attn, figsize=self.figsize, cmap=self.cmap, cbar_range=(low, high)
        )

    def log_wandb_image(
        self, tag: str, it: int, fig: Figure, caption: Optional[str] = None
    ):
        image = wandb.Image(fig, caption=caption if self.use_caption else None)
        if self.run is not None:
            self.run.log({tag: image, "step": it})
        else:
            print(tag, caption)
            image.image.show()
        plt.close(fig)

    def log_images(self, buffer: List[tuple]):
        for args, kwargs in buffer:
            self.log_wandb_image(*args, **kwargs)


@dataclass
class WandbLogger2D(WandbLogger):
    use_caption: bool = field(init=False, default=False)

    def pre_call_hook(self, *tensors: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
        # BCWH -> 1CWHB (D is dummy dimension simulated by B)
        return [t if t is None else t[..., None].swapaxes(0, -1) for t in tensors]
