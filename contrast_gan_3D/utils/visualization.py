from collections import defaultdict
from typing import Iterable, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm

from contrast_gan_3D import config
from contrast_gan_3D.alias import Array
from contrast_gan_3D.constants import VMAX, VMIN
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


# NOTE assumes channel-last images
def plot_axial_slices(
    slices: np.ndarray,
    axes: Optional[Union[np.ndarray, Axes]] = None,
    tight: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> np.ndarray:
    if len(slices.shape) < 2:
        slices = slices[..., None]

    if axes is None:
        _, axes = plt.subplots(*compute_grid_size(slices.shape[-1]), figsize=figsize)
    axes = ensure_2D_axes(axes)

    for i, ax in enumerate(axes.flat):
        if i < slices.shape[-1]:
            ax.imshow(slices[..., i].T, cmap="gray", **kwargs)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_visible(False)

    if title is not None:
        axes[0, 0].get_figure().suptitle(title)

    if tight:
        plt.tight_layout()

    return axes


# NOTE assumes channel-last images and n/centerlines in image coordinates
def plot_axial_centerlines(
    image: Union[np.ndarray, h5py.Dataset],
    centerlines: np.ndarray,
    n: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    axes: Optional[Union[np.ndarray, Axes]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    if isinstance(centerlines, np.ndarray):
        cond = centerlines.dtype.kind == "i"
    else:
        cond = not torch.is_floating_point(centerlines)
    assert cond, "Centerlines should be given in image coordinates"

    if n is None:
        n = len(centerlines)
    assert n <= len(centerlines), f"Cannot plot {n}/{len(centerlines)} centerlines!"
    if n > 1000:
        logger.info(f"Reducing centerline sample size from {n} to 100")
        n = 1000

    if n == len(centerlines):
        chosen_ctls = centerlines
    else:
        if rng is None:
            rng = np.random.default_rng(seed=config.DEFAULT_SEED)
        chosen_ctls = rng.choice(centerlines, n)
    chosen_ctls.sort(axis=0)

    axes = plot_axial_slices(
        image[..., np.unique(chosen_ctls[..., 2])], axes=axes, tight=False, **kwargs
    )

    unique_slices_ctls = defaultdict(list)
    for ctl in chosen_ctls:
        unique_slices_ctls[ctl[2]].append(ctl[:2])
    for slice_idx, ctls in unique_slices_ctls.items():
        unique_slices_ctls[slice_idx] = np.vstack(ctls)

    n_empty_ax = len(axes.flat) - len(unique_slices_ctls)
    pad_slice_ctls = list(unique_slices_ctls) + [None] * n_empty_ax

    for slice_idx, ax in zip(pad_slice_ctls, axes.flat):
        if slice_idx is None:
            ax.set_visible(False)
        else:
            slice_ctls = unique_slices_ctls[slice_idx]
            ax.scatter(slice_ctls[:, 0], slice_ctls[:, 1], c="red", edgecolors="black")
            ctls_coords = tuple(slice_ctls.flat) + (int(slice_idx),)
            if len(ctls_coords) > 3:
                # many centerlines on same slice: only give centerlines' z coord
                ctls_coords = ctls_coords[-1]
            ax.set_title(f"{ctls_coords}, {len(slice_ctls)}")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    full_title = f"{len(chosen_ctls)}/{len(centerlines)} centerlines"
    if title is not None:
        full_title = f"{title} {full_title}"
    axes[0, 0].get_figure().suptitle(full_title)
    plt.tight_layout()

    return axes


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
    # order: axial, sagittal, coronal
    for ax, patch in zip(
        axes.flat, [ostium_patch[..., z], ostium_patch[:, y, :], ostium_patch[x, ...]]
    ):
        ax.imshow(patch.T, **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return axes


# NOTE works best with LPS orientation
def plot_mid_slice(
    image: np.ndarray,
    axes: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    tight: bool = True,
    vmin: int = VMIN,
    vmax: int = VMAX,
) -> np.ndarray:
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes = ensure_2D_axes(axes)

    args = dict(zip(["cmap", "vmin", "vmax"], ["gray", vmin, vmax]))
    middle_x, middle_y, middle_z = image.shape // np.array(2)

    axes[0, 0].imshow(image[..., middle_z].T, **args)
    axes[0, 0].set_title("Axial")
    axes[0, 1].imshow(np.flip(image[middle_x, ...].T, 0), **args)
    axes[0, 1].set_title("Sagittal")
    axes[0, 2].imshow(np.flip(image[:, middle_y, :].T, 0), **args)
    axes[0, 2].set_title("Coronal")

    full_title = f"{tuple(image.shape)}, middle: {(middle_x, middle_y, middle_z)}"
    if title is not None:
        full_title = f"{title} {full_title}"
    axes[0, 0].get_figure().suptitle(full_title)

    if tight:
        plt.tight_layout()

    return axes


def plot_gmm_fitted_ostium_patch(
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


# NOTE possible improvements:
# - draw a semi-transparent box for patch volume extent
# - use plotly
def plot_extracted_patch_3D(
    centerlines_mask: Array,
    extracted_patch: Array,
    patch_center: Array,
    ostia: Optional[Array] = None,
    figsize: Tuple[int, int] = (10, 5),
    axes: Optional[np.ndarray] = None,
) -> np.ndarray:
    if axes is None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    axes = ensure_2D_axes([ax1, ax2])

    for i, ax in enumerate(axes.flat):
        assert ax.name == "3d", f"Axis {i} does not support 3D plotting"

    patch_plot_c = ("purple", 1)
    point_s = 60

    whole_image_patch_mask = geom.expand_3D_patch_whole_image(
        extracted_patch, centerlines_mask.shape, extracted_patch.shape, patch_center
    )

    plot_centerlines_3D(
        geom.grid_to_cartesian_coords(centerlines_mask),
        color=("C1", 0.1),
        ax=ax1,
        depthshade=False,
    )
    plot_centerlines_3D(
        geom.grid_to_cartesian_coords(whole_image_patch_mask),
        color=patch_plot_c,
        ax=ax1,
        depthshade=False,
    )
    plot_centerlines_3D(
        patch_center[None, ...], color=("black", 1), ax=ax1, depthshade=False, s=point_s
    )
    if ostia is not None:
        # NOTE idky plotting both in one call only plots one ostium
        for ostium in ostia:
            plot_centerlines_3D(
                ostium[None], color=patch_plot_c, ax=ax1, depthshade=False, s=point_s
            )

    plot_centerlines_3D(
        geom.grid_to_cartesian_coords(extracted_patch),
        ax=ax2,
        color=patch_plot_c,
    )
    plot_centerlines_3D(
        np.array([extracted_patch.shape]) // 2,
        color=("black", 1),
        ax=ax2,
        depthshade=False,
        s=point_s,
    )

    return axes
