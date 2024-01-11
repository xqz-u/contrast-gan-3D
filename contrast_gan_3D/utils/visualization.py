from collections import defaultdict
from typing import Optional, Tuple, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from contrast_gan_3D import config


def compute_grid_size(n: int) -> Tuple[int, int]:
    rows = int(round(np.sqrt(n)))
    return rows, int(np.ceil(n / rows))


def plot_centerlines_3D(
    centerlines: np.ndarray,
    title: str = "Centerlines",
    downsample_factor: int = 1,
    ax: Optional[Axes] = None,
) -> Axes:
    assert centerlines.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

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
) -> np.ndarray:
    if len(slices.shape) < 2:
        slices = slices[..., None]

    if axes is None:
        _, axes = plt.subplots(*compute_grid_size(slices.shape[-1]), figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, ax in enumerate(axes.flat):
        if i < slices.shape[-1]:
            ax.imshow(slices[..., i], cmap="gray")
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
    ), "Centerlines must be given in image coordinates"

    if n is None:
        n = len(centerlines)
    assert n <= len(centerlines), f"Cannot plot {n}/{len(centerlines)} centerlines!"
    if n > 1000:
        print(f"Reducing centerline sample size from {n} to 100")
        n = 1000

    if rng is None:
        rng = np.random.default_rng(seed=config.DEFAULT_SEED)

    chosen_ctls = np.sort(rng.choice(centerlines, n), axis=0)

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
            ax.set_title(f"{slice_idx}, {len(slice_ctls)}")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    full_title = f"{len(chosen_ctls)}/{len(centerlines)} centerlines"
    if title is not None:
        full_title = f"{title} {full_title}"
    axes[0, 0].get_figure().suptitle(full_title)
    plt.tight_layout()

    return axes
