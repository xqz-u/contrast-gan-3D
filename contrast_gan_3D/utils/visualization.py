from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import seaborn as sns
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm
from torchvision.utils import make_grid

from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.utils import downsample
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import logging_utils

logger = logging_utils.create_logger(name=__name__)


def ensure_2D_axes(axes: Union[np.ndarray, Axes]) -> np.ndarray:
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    if len(axes.shape) < 2:
        axes = axes[None, ...]
    return axes


def compute_grid_size(n: int) -> Tuple[int, int]:
    rows = int(round(np.sqrt(n)))
    return rows, int(np.ceil(n / rows))


def plot_centerlines_3D(
    centerlines: np.ndarray,
    downsample_factor: int = 1,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    **scatter_kwargs,
) -> Axes:
    assert centerlines.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    assert ax.name == "3d", "Axis does not support 3D plotting"

    centerlines = centerlines[::downsample_factor]
    ax.scatter(
        centerlines[:, 0], centerlines[:, 1], centerlines[:, 2], **scatter_kwargs
    )

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


# NOTE assumes `slices` of shape CWH(B)
def plot_axial_slices(
    slices: Union[torch.Tensor, np.ndarray],
    fig: Optional[Figure] = None,
    tight: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = "gray",
    cbar_range: Tuple[int, int] | None = None,
    **grid_args,
) -> Figure:
    if isinstance(slices, np.ndarray):
        slices = torch.from_numpy(slices)
    if len(slices.shape) < 4:
        slices = slices[..., None]  # virtual batch dimension
    # CWHB -> BCHW -> C,HxB,WxB
    grid = make_grid(slices.permute(3, 0, 2, 1).to(float), **grid_args)
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax, *_ = fig.axes
    ax.imshow(grid.permute(1, 2, 0), cmap=cmap)
    # manually add colorbar with explicit range
    if cbar_range is not None:
        norm = colors.Normalize(*cbar_range)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
        cbar.set_ticks(np.linspace(*cbar_range, 10, dtype=int))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if tight:
        fig.tight_layout()
    return fig


def plot_axial_slices_plus_centerlines(
    slices: np.ndarray | torch.Tensor,
    ctls: np.ndarray | torch.Tensor | None = None,
    fig: Figure | None = None,
    figsize: tuple[int, int] = (10, 10),
    **args,
) -> Figure:
    assert len(slices.shape) >= 3
    if len(slices.shape) < 4:
        slices = slices[None]  # add channel dim

    fig = plot_axial_slices(slices, fig=fig, figsize=figsize, **args)

    if ctls is not None:
        assert len(ctls.shape) >= 3
        if len(ctls.shape) < 4:
            ctls = ctls[None]  # add channel dim
        if not isinstance(ctls, torch.Tensor):
            ctls = torch.from_numpy(ctls)

        # show centerlines by scattering manually
        ctls = ctls.permute(3, 0, 2, 1).to(float)  # CWHD -> DCHW
        ctls_cart = geom.grid_to_cartesian_coords(make_grid(ctls))  # DHW (zyx)
        fig.axes[0].scatter(
            ctls_cart[:, 2],
            ctls_cart[:, 1],
            c="red",
            s=plt.rcParams["lines.markersize"] * 0.8,
        )

    return fig


# NOTE first dimension: batch size
def plot_image_histogram(
    *images_bw,
    tight: bool = False,
    axes: Optional[Union[Axes, np.ndarray]] = None,
    figsize: Tuple[int, int] = (10, 5),
    **hist_kwargs,
) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(*compute_grid_size(len(images_bw)), figsize=figsize)
    axes = ensure_2D_axes(axes)

    imgs_pad = [*images_bw] + [None] * (len(axes.flat) - len(images_bw))

    for img, ax in zip(imgs_pad, axes.flat):
        if img is None:
            ax.set_visible(False)
        else:
            if isinstance(img, tuple) and len(img) == 2:
                img, title = img
                ax.set_title(title)
            ax.hist(img.ravel(), color="black", bins=80, **hist_kwargs)

    if tight:
        plt.tight_layout()
    return axes


# NOTE works with LPS orientation. image: WHD, return order: axial, sagittal, coronal
def get_medical_views(scan: np.ndarray, xyz: np.ndarray) -> list[np.ndarray]:
    x, y, z = xyz
    return [scan[..., z].T, np.flip(scan[x, ...].T, 0), np.flip(scan[:, y, :].T, 0)]


def plot_ostium_patch(
    ostium_patch: np.ndarray,
    coords: Union[Iterable[int], str] = "middle",
    axes: Optional[Union[np.ndarray, Axes]] = None,
    vmin: int = VMIN,
    vmax: int = VMAX,
    title: Optional[str] = None,
) -> np.ndarray:
    if isinstance(coords, str):
        assert coords == "middle"
        x, y, z = np.array(ostium_patch.shape) // 2
    else:
        x, y, z = coords

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(7, 5))
    axes = ensure_2D_axes(axes)

    if title is not None:
        axes[0, 0].get_figure().suptitle(title)

    kwargs = dict(zip(["vmin", "vmax", "cmap"], [vmin, vmax, "gray"]))
    views = get_medical_views(ostium_patch, np.array([x, y, z]))
    for ax, patch in zip(axes.flat, views):
        ax.imshow(patch, **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return axes


def plot_mid_slice(
    image: np.ndarray,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    vmin: int = VMIN,
    vmax: int = VMAX,
) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 5))

    args = dict(zip(["cmap", "vmin", "vmax"], ["gray", vmin, vmax]))
    middle = image.shape // np.array(2)

    views = get_medical_views(image, middle)
    for ax, ax_title, view in zip(axes.flat, ["Axial", "Sagittal", "Coronal"], views):
        ax.imshow(view, **args)
        ax.set_title(ax_title)

    full_title = f"{tuple(image.shape)}, middle: {middle}"
    if title is not None:
        full_title = f"{title} {full_title}"
    axes[0].get_figure().suptitle(full_title)

    return axes


def plot_GMM_fitted_ostium_patch(
    ostium_patch: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    plot_ostia_kwargs: Optional[dict] = None,
) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(8, 5))

    plot_ostia_kwargs = plot_ostia_kwargs or {}
    plot_ostium_patch(ostium_patch, axes=axes[..., :-1], **plot_ostia_kwargs)

    ax = axes.flat[-1]
    if title is not None:
        ax.set_title(title)

    ax.hist(ostium_patch.reshape(-1, 1), density=True, color="black", bins=80)

    x = np.arange(-300, 900, 10)  # HU range of interest
    y = norm.pdf(x, mean, std) * weights
    # total cumulative probability
    ax.plot(x, y.sum(0), lw=3, c=f"C{n_components}", ls="dashed")
    for i, yy in enumerate(y):
        ax.plot(x, yy, lw=3, c=f"C{i}")

    return axes


def downsampled_hist(
    a: np.ndarray,
    maxsize: int = int(1e5),
    xlims: tuple[int, int] | None = None,
    **kwargs,
) -> Axes:
    if len(a) > maxsize:
        a = downsample(a, maxsize)
    if xlims is not None:
        a = np.concatenate([np.array([xlims[0]]), a, np.array([xlims[1]])])
    return sns.histplot(a, **kwargs)


def plot_HU_distributions(
    subopt: np.ndarray,
    corrected_subopt: np.ndarray,
    opt: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    alpha: float = 0.6,
    **kwargs,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    args = {"stat": "density", "kde": True, "edgecolor": "none", "alpha": alpha}

    downsampled_hist(subopt, label="Suboptimal", ax=ax, **args, **kwargs)
    downsampled_hist(
        corrected_subopt, label="Corrected suboptimal", ax=ax, **args, **kwargs
    )
    downsampled_hist(opt, label="Optimal", ax=ax, **args, **kwargs)

    if title is not None:
        ax.set_title(title)

    return ax


def HU_distribution_shift_plot(
    og_voxels: dict[ScanType, dict[str, np.ndarray]],
    corrected_voxels: dict[ScanType, dict[str, np.ndarray]],
    plot_savepath: str | Path | None,
    show: bool,
    save_suffix: str = "png",
    titles_map: dict[str, str] | None = None,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].set_ylabel(r"$\leq$ 300 HU")
    axes[1, 0].set_ylabel(r"$\geq$ 500 HU")
    xlims = (-200, 800)
    for ax in axes.flat:
        ax.set_xlim(*xlims)
        # ax.set_ylim(0, 0.0075)
        ax.set_yticks([])

    for i, st in enumerate([ScanType.LOW, ScanType.HIGH]):
        for j, tag in enumerate(og_voxels[st].keys()):
            ax_title = tag.capitalize()
            if i == 1:
                ax_title = None
            elif titles_map is not None:
                ax_title = titles_map.get(tag, ax_title)
            plot_HU_distributions(
                og_voxels[st][tag],
                corrected_voxels[st][tag],
                og_voxels[ScanType.OPT][tag],
                ax=axes[i, j],
                title=ax_title,
                # xlims=xlims,
            )
        # for ax in axes[i]:
        #     if i == 0:
        #         ax.set_xticks([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    if plot_savepath is not None:
        plot_savepath = Path(plot_savepath).with_suffix(save_suffix)
        fig.savefig(plot_savepath, bbox_inches="tight")
        print(f"Saved figure to {str(plot_savepath)!r}")
    if show:
        plt.show()
    plt.close(fig)
