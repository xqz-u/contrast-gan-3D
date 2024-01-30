from typing import Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch

from contrast_gan_3D import constants
from contrast_gan_3D.alias import Array, Shape3D
from contrast_gan_3D.utils import array, io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


def check_3D_arrays(*arrays: Tuple[Array]):
    assert all(el.shape[-1] == 3 for el in arrays)


def world_to_image_coords(world_coords: Array, offset: Array, spacing: Array) -> Array:
    if isinstance(spacing, tuple):  # from torchio.Subject.spacing
        spacing = torch.tensor(spacing)
    check_3D_arrays(world_coords, offset, spacing)
    ret = ((world_coords - offset) / spacing).round()
    return ret.astype(int) if isinstance(ret, np.ndarray) else ret.to(int)


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
    coords_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    image = io_utils.ensure_HU_intensities(image)

    ostia_rows = ostia_df[ostia_df["ID"] == image_id]
    if len(ostia_rows) != 2:
        logger.debug(
            "Something's off with '%s' ostia, shape: %s", image_id, ostia_rows.shape
        )

    coords_indexer = [f"{coords_prefix}{c}" for c in list("xyz")]
    ostia_world_coords = ostia_rows[coords_indexer].values
    ostia_coords = ostia_world_coords - meta["offset"]
    ostia_patch_samples = [
        draw_sample_3D_world_fast(
            image, *coords, meta["spacing"], patch_size, patch_spacing
        )
        for coords in ostia_coords
    ]

    return np.stack(ostia_patch_samples), ostia_world_coords


# NOTE xyz is the center of the patch
def extract_3D_patch(
    img: Union[Array, h5py.Dataset], size: Shape3D, xyz: np.ndarray
) -> Tuple[Array, Array]:
    half = array.parse_patch_size(size, img.shape) // 2
    bbox = np.dstack([xyz - half, xyz + half]).squeeze()
    indexer = [slice(*box) for box in bbox]
    # shape: `size`, possibly < `img.shape`
    patch = img[*indexer]
    # shape: `img.shape` - mask of extracted coordinates in original array
    patch_mask = np.zeros_like(img)
    patch_mask[*indexer] = patch

    return patch, patch_mask


def extract_random_3D_patch(
    img: Union[Array, h5py.Dataset],
    size: Shape3D,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    size = array.parse_patch_size(size, img.shape)
    # xyz is the *center* of the extracted cube
    xyz = [
        rng.integers(extent, dim_high - extent + 1)
        for dim_high, extent in zip(img.shape, size // 2)
    ]
    xyz = np.array(xyz)
    return *extract_3D_patch(img, size, xyz), xyz
