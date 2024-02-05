import numpy as np

from contrast_gan_3D.alias import Shape3D


def parse_patch_size(target_shape: Shape3D, input_shape: Shape3D) -> np.ndarray:
    target_shape = np.array(target_shape)
    for i, dim in enumerate(target_shape):
        if dim == -1:
            target_shape[i] = input_shape[i]
    return target_shape


