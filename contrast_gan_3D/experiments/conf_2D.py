from batchgenerators.transforms.spatial_transforms import MirrorTransform

from contrast_gan_3D.experiments.basic_conf import *

train_patch_size = train_patch_size[:-1]
val_patch_size = (512, 512)

train_transform_args = {
    "patch_size": train_patch_size,
    "random_crop": False,
    # deformation
    "do_elastic_deform": False,
    # scaling
    "do_scale": False,
    # rotation
    "do_rotation": True,
    **{
        f"angle_{ax}": (-geom.deg_to_radians(360), geom.deg_to_radians(360))
        for ax in list("xyz")
    },
    "p_rot_per_sample": 0.5,
}
train_transform = Compose(
    [
        SpatialTransform_2(**train_transform_args),
        MirrorTransform(axes=(0, 1), p_per_sample=0.5),
        NumpyToTensor(keys=["data"]),
        NumpyToTensor(keys=["seg"], cast_to="bool"),
    ]
)
