from batchgenerators.transforms.spatial_transforms import MirrorTransform

from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.trainer.logger.WandbLogger import WandbLogger2D

train_patch_size = train_patch_size[:-1]
val_patch_size = (512, 512)

train_batch_size = {
    v.value: b
    for v, b in [(ScanType.OPT, 256), (ScanType.LOW, 128), (ScanType.HIGH, 128)]
}
val_batch_size = train_batch_size.copy()

generator_args.update(is_2D=True, n_resnet_blocks=6)
generator_class = partial(ResnetGenerator, **generator_args)

critic_args.update(is_2D=True, init_channels_out=16)
critic_class = partial(PatchGANDiscriminator, **critic_args)

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

logger_interface = WandbLogger2D(scaler, rng=rng, figsize=(10, 10))
logger_interface = MultiThreadedLogger(logger_interface)
