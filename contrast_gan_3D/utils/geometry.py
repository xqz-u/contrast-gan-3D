import numpy as np
import torch

from contrast_gan_3D import utils
from contrast_gan_3D.alias import Array, ArrayShape, Shape3D
from contrast_gan_3D.constants import AORTIC_ROOT_PATCH_SIZE, AORTIC_ROOT_PATCH_SPACING
from contrast_gan_3D.utils import logging_utils

logger = logging_utils.create_logger(name=__name__)


def check_3D_arrays(*arrays: tuple[Array]):
    for el in arrays:
        assert el.shape[-1] == 3, el.shape


def deg_to_radians(deg: float) -> float:
    return deg * np.pi / 180


def world_to_image_coords(world_coords: Array, offset: Array, spacing: Array) -> Array:
    if isinstance(spacing, tuple):  # from torchio.Subject.spacing
        spacing = torch.tensor(spacing)
    check_3D_arrays(world_coords, offset, spacing)
    ret = ((world_coords - offset) / spacing).round()
    return ret.astype(int) if isinstance(ret, np.ndarray) else ret.to(int)


# NOTE previous codebase
def fast_trilinear(
    input_array: np.ndarray,
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
    image: np.ndarray,
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
    scan: np.ndarray,
    ostia_world: np.ndarray,
    offset: np.ndarray,
    spacing: np.ndarray,
    patch_size: np.ndarray = AORTIC_ROOT_PATCH_SIZE,
    patch_spacing: np.ndarray = AORTIC_ROOT_PATCH_SPACING,
) -> np.ndarray:
    ostia_patch_samples = [
        draw_sample_3D_world_fast(scan, *coords, spacing, patch_size, patch_spacing)
        for coords in ostia_world - offset
    ]
    return np.stack(ostia_patch_samples)


def ensure_valid_bounds(s: int, e: int, target_size: int, size: int) -> tuple[int, int]:
    assert not (s < 0 and e > size), f"{target_size} < {size}"
    if s < 0:
        s, e = 0, target_size
    if e > size:
        s, e = size - target_size, size
    return s, e


# NOTE vectorized version missing :/
def ensure_valid_bounds_arr(
    bounds: np.ndarray, target_shape: ArrayShape, shape: ArrayShape
):
    for (i, (s, e)), target_size, size in zip(enumerate(bounds), target_shape, shape):
        bounds[i] = ensure_valid_bounds(s, e, target_size, size)


def get_patch_bounds(
    target_shape: ArrayShape, source_shape: ArrayShape, coords: np.ndarray
) -> np.ndarray:
    half = utils.parse_patch_size(target_shape, source_shape) // 2
    target_shape = np.array(target_shape)
    bbox = np.dstack([coords - half, coords + half + target_shape % 2]).squeeze()
    ensure_valid_bounds_arr(bbox, target_shape, source_shape)
    return bbox


def world_to_grid_coords(
    centerlines: np.ndarray,
    offset: np.ndarray,
    spacing: np.ndarray,
    grid_shape: Shape3D,
) -> np.ndarray:
    centerlines_img_coords = world_to_image_coords(centerlines, offset, spacing)
    # NOTE many centerlines overlap once mapped to image coordinates
    centerlines_img_coords = np.unique(centerlines_img_coords, axis=0)
    centerlines_grid = np.zeros(grid_shape, dtype=np.uint8)
    # clip the coordinates within array bounds
    clipped_coords = [
        np.clip(centerlines_img_coords[:, i], 0, centerlines_grid.shape[i] - 1)
        for i in range(3)
    ]
    centerlines_grid[*clipped_coords] = 1
    return centerlines_grid


def grid_to_cartesian_coords(grid_mask_3D: Array) -> Array:
    cart_coords = np.dstack(np.where(grid_mask_3D)).squeeze()
    if isinstance(grid_mask_3D, torch.Tensor):
        cart_coords = torch.tensor(cart_coords)
    return cart_coords
