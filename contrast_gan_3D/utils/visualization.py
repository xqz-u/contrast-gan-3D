from collections import defaultdict
from typing import Iterable, Optional, Tuple, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from contrast_gan_3D import config
from contrast_gan_3D.constants import VMAX, VMIN
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
    title: str = "Centerlines",
    downsample_factor: int = 1,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> Axes:
    assert centerlines.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    assert ax.name == "3d", "Axis does not support 3D plotting"

    centerlines = centerlines[::downsample_factor]
    ax.scatter(centerlines[:, 0], centerlines[:, 1], centerlines[:, 2])

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
    figsize: Tuple[int, int] = (20, 15),
    vmin: Optional[int] = None,
    vmax: Optional[int] = None,
) -> np.ndarray:
    if len(slices.shape) < 2:
        slices = slices[..., None]

    if axes is None:
        _, axes = plt.subplots(*compute_grid_size(slices.shape[-1]), figsize=figsize)
    axes = ensure_2D_axes(axes)

    for i, ax in enumerate(axes.flat):
        if i < slices.shape[-1]:
            ax.imshow(slices[..., i].T, cmap="gray", vmin=vmin, vmax=vmax)
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
    assert (
        centerlines.dtype.kind == "i"
    ), "Centerlines should be given in image coordinates"

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
            ctls_coords = tuple(slice_ctls.flat) + (slice_idx,)
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


def plot_ostia_patch(
    ostia_patch: np.ndarray,
    coords: Union[Iterable[int], str] = "middle",
    axes: Optional[Union[np.ndarray, Axes]] = None,
    vmin: int = VMIN,
    vmax: int = VMAX,
    title: Optional[str] = None,
) -> np.ndarray:
    if isinstance(coords, str):
        assert coords == "middle"
        x, y, z = np.array(ostia_patch.shape[1:]) // 2
    else:
        x, y, z = coords

    if axes is None:
        _, axes = plt.subplots(2, 3, figsize=(10, 5))

    if title is not None:
        axes[0, 0].get_figure().suptitle(title)

    kwargs = dict(zip(["vmin", "vmax", "cmap"], [vmin, vmax, "gray"]))
    for i in range(2):
        # axial
        axes[i, 0].imshow(ostia_patch[i, ..., z], **kwargs)
        # sagittal
        axes[i, 1].imshow(ostia_patch[i, :, y, :], **kwargs)
        # coronal
        axes[i, 2].imshow(ostia_patch[i, x, ...], **kwargs)

    return axes


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
