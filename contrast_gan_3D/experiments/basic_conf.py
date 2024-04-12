from functools import partial

import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from contrast_gan_3D.constants import MAX_HU, MIN_HU, TRAIN_PATCH_SIZE, VAL_PATCH_SIZE
from contrast_gan_3D.data.Scaler import FactorZeroCenterScaler
from contrast_gan_3D.model.discriminator import PatchGANDiscriminator
from contrast_gan_3D.model.generator import ResnetGenerator
from contrast_gan_3D.trainer.logger.LoggerInterface import MultiThreadedLogger
from contrast_gan_3D.trainer.logger.WandbLogger import WandbLogger
from contrast_gan_3D.utils import geometry as geom

train_iterations = int(1e4)
val_iterations = 10
train_generator_every = 5  # from WGAN paper
# seed = DEFAULT_SEED
seed = None
checkpoint_every = int(1e3)
validate_every = 400
log_every = 100
log_images_every = 500

# ------------ MODEL ------------
lr = 2e-4
betas = (5e-1, 0.999)
milestones = list(map(int, [6e3, 8e3]))
lr_gamma = 0.1
weight_clip = 0.01  # from WGAN paper

# HU loss & details
max_HU_delta = 600
desired_HU_bounds = (350, 450)
HU_norm_range = (MIN_HU, MAX_HU)
scaler = FactorZeroCenterScaler(*HU_norm_range, max_HU_delta)

logger_interface = WandbLogger(scaler, rng=np.random.default_rng(seed=seed))
logger_interface = MultiThreadedLogger(logger_interface)

generator_args = {
    "n_resnet_blocks": 4,
    "n_updownsample_blocks": 2,
    "init_channels_out": 16,
}
generator_class = partial(ResnetGenerator, **generator_args)
generator_optim_class = partial(Adam, lr=lr, betas=betas)
generator_lr_scheduler_class = partial(
    MultiStepLR, milestones=milestones, gamma=lr_gamma
)

critic_args = {
    "channels_in": 1,
    "init_channels_out": 8,
    "discriminator_depth": 3,
}
critic_class = partial(PatchGANDiscriminator, **critic_args)
critic_optim_class = partial(Adam, lr=lr, betas=betas)
critic_lr_scheduler_class = partial(MultiStepLR, milestones=milestones, gamma=lr_gamma)

# ------------ DATA ------------
cval_folds = 5

train_patch_size = TRAIN_PATCH_SIZE
train_batch_size = 6  # 12 subopt 6 opt
# train_batch_size = 3

val_patch_size = VAL_PATCH_SIZE
val_batch_size = 3  # 6 subopt 3 opt
# val_batch_size = 2  # 6 subopt 3 opt

num_workers = (train_batch_size * 2, val_batch_size * 2)  # (train, validation)

dataset_paths = ["/home/marco/data/ostia_final.xlsx"]
train_transform_args = {
    "patch_size": train_patch_size,
    "random_crop": False,
    # deformation
    "do_elastic_deform": True,
    "deformation_scale": (0, 0.25),  # default
    "p_el_per_sample": 0.1,
    # scaling
    "do_scale": True,
    "scale": (0.7, 1.4),  # default
    "p_scale_per_sample": 0.2,
    # rotation
    "do_rotation": True,
    **{
        f"angle_{ax}": (-geom.deg_to_radians(30), geom.deg_to_radians(30))
        for ax in list("xyz")
    },
    "p_rot_per_sample": 0.2,
}
train_transform = Compose(
    [
        SpatialTransform_2(**train_transform_args),
        NumpyToTensor(keys=["data"]),
        NumpyToTensor(keys=["seg"], cast_to="bool"),
    ]
)
