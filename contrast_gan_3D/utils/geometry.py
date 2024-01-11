import numpy as np


def check_3D_arrays(*arrays):
    assert all(el.shape[-1] == 3 for el in arrays)


def world_to_image_coords(
    world_coords: np.ndarray, offset: np.ndarray, spacing: np.ndarray
) -> np.ndarray:
    check_3D_arrays(world_coords, offset, spacing)
    return ((world_coords - offset) / spacing).round().astype(int)
