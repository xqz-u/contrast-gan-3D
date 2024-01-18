from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
import pandas as pd

from contrast_gan_3D import constants
from contrast_gan_3D.utils import io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


def check_3D_arrays(*arrays):
    assert all(el.shape[-1] == 3 for el in arrays)


def world_to_image_coords(
    world_coords: np.ndarray, offset: np.ndarray, spacing: np.ndarray
) -> np.ndarray:
    check_3D_arrays(world_coords, offset, spacing)
    return ((world_coords - offset) / spacing).round().astype(int)


# NOTE previous codebase
def fast_trilinear(
    input_array: Union[np.ndarray, h5py.Dataset],
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    z_indices: np.ndarray,
):
    x0 = x_indices.astype(np.int64)
    y0 = y_indices.astype(np.int64)
    z0 = z_indices.astype(np.int64)

    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # boundary checks
    for i, arr in enumerate([x0, y0, z0, x1, y1, z1]):
        limit = input_array.shape[i % 3]
        arr[arr >= limit] = limit - 1
        arr[arr < 0] = 0

    x, y, z = x_indices - x0, y_indices - y0, z_indices - z0

    return (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )


# NOTE previous codebase
# NOTE as the name suggests: x,y,z are in **world coordinates**
def draw_sample_3D_world_fast(
    image: Union[np.ndarray, h5py.Dataset],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    imagespacing: np.ndarray,
    patchsize: np.ndarray,
    patchspacing: np.ndarray,
):
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]

    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])

    coords = np.concatenate(
        (
            np.reshape(xs, (1, xs.shape[0])),
            np.reshape(ys, (1, ys.shape[0])),
            np.reshape(zs, (1, zs.shape[0])),
            np.zeros((1, xs.shape[0]), dtype=np.float32),
        ),
        axis=0,
    )

    xs = np.squeeze(coords[0, :]) + (x / imagespacing[0])
    ys = np.squeeze(coords[1, :]) + (y / imagespacing[1])
    zs = np.squeeze(coords[2, :]) + (z / imagespacing[2])

    return fast_trilinear(image, xs, ys, zs).reshape(patchsize)


def extract_ostia_patch_3D(
    image: np.ndarray,
    meta: dict,
    image_id: str,
    ostia_df: pd.DataFrame,
    patch_size: np.ndarray = constants.AORTIC_ROOT_PATCH_SIZE,
    patch_spacing: np.ndarray = constants.AORTIC_ROOT_PATCH_SPACING,
    is_cadrads: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    image = io_utils.ensure_HU_range(image)

    datapoint_key = "ID" if is_cadrads else "id"
    ostia_rows = ostia_df[ostia_df[datapoint_key] == image_id]
    if len(ostia_rows) != 2:
        logger.debug(
            "Something's off with '%s' ostia, shape: %s", image_id, ostia_rows.shape
        )

    coords_indexer = [f"{'ostium_' if is_cadrads else ''}{c}" for c in list("xyz")]
    ostia_world_coords = ostia_rows[coords_indexer].values
    ostia_coords = ostia_world_coords - meta["offset"]
    ostia_patch_samples = [
        draw_sample_3D_world_fast(
            image, *coords, meta["spacing"], patch_size, patch_spacing
        )
        for coords in ostia_coords
    ]

    return np.stack(ostia_patch_samples), ostia_world_coords
