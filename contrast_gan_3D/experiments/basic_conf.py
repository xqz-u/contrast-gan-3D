import torch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from contrast_gan_3D.constants import DEFAULT_SEED, TRAIN_PATCH_SIZE
from contrast_gan_3D.model.discriminator import PatchGAN
from contrast_gan_3D.model.generator import ResnetGenerator
from contrast_gan_3D.utils import geometry as geom

# **** NOTE **** change GPU index from here
device_str = "cpu"
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device_str = f"cuda:{torch.cuda.current_device()}"
device = torch.device(device_str)

train_iterations = int(1e4)
val_iterations = 20
train_generator_every = 5
seed = DEFAULT_SEED
fold_idx = 0
checkpoint_every = int(1e3)
validate_every = 100
log_every = 50

# ------------ MODEL ------------
lr = 2e-4
betas = (5e-1, 0.999)
milestones = list(map(int, [6e3, 8e3]))
lr_gamma = 0.1
max_HU_diff = 600
HULoss_args = {"HU_diff": max_HU_diff, "min_HU": 350, "max_HU": 450}

generator_args = {
    "n_resnet_blocks": 6,
    "n_updownsample_blocks": 2,
    "n_feature_maps": 16,
}
generator = ResnetGenerator(**generator_args).to(device)

generator_optim = Adam(generator.parameters(), lr=lr, betas=betas)
generator_lr_scheduler = MultiStepLR(
    generator_optim, milestones=milestones, gamma=lr_gamma
)

discriminator_args = {"discriminator_depth": 3, "n_feature_maps": 16}
discriminator = PatchGAN(1, 1, **discriminator_args).to(device)

discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=betas)
discriminator_lr_scheduler = MultiStepLR(
    discriminator_optim, milestones=milestones, gamma=lr_gamma
)

# ------------ DATA ------------
train_patch_size = TRAIN_PATCH_SIZE
train_batch_size = 6  # 12 subopt 6 opt
val_patch_size = (256, 256, 128)
val_batch_size = 3  # 6 subopt 3 opt
num_workers = (12, 6)  # (train, validation)
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
