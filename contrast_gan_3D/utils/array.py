import numpy as np
import torch

from contrast_gan_3D.alias import Array, Shape3D


def parse_patch_size(target_shape: Shape3D, input_shape: Shape3D) -> np.ndarray:
    target_shape = np.array(target_shape)
    for i, dim in enumerate(target_shape):
        if dim == -1:
            target_shape[i] = input_shape[i]
    return target_shape


def grid_mask_to_cartesian_3D(grid_mask_3D: Array) -> Array:
    cart_coords = np.dstack(np.where(grid_mask_3D)).squeeze()
    if isinstance(grid_mask_3D, torch.Tensor):
        cart_coords = torch.tensor(cart_coords)
    return cart_coords
