import torch
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from contrast_gan_3D.constants import DEFAULT_SEED, TRAIN_PATCH_SIZE
from contrast_gan_3D.model.discriminator import NLayerDiscriminator
from contrast_gan_3D.model.generator import ResnetGenerator
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import object_name

# NOTE **** change device number from here ****
# drawback of instantiating everything outside Trainer. Could set
# CUDA_VISIBLE_DEVICES read from cl, but needs to be done before importing torch
# globally
GPU = 4
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

train_iterations = 10
train_generator_every = 2
seed = DEFAULT_SEED
fold_idx = 0
checkpoint_every = 2
validate_every = 2

# ------------ MODEL ------------
lr = 2e-4
betas = (5e-1, 0.999)
milestones = [3, 7]
lr_gamma = 0.1
HULoss_args = {"bias": -1024, "factor": 600, "min_HU": 350, "max_HU": 450}

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
discriminator = NLayerDiscriminator(1, 1, **discriminator_args).to(device)

discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=betas)
discriminator_lr_scheduler = MultiStepLR(
    discriminator_optim, milestones=milestones, gamma=lr_gamma
)

# ------------ DATA ------------
train_patch_size = TRAIN_PATCH_SIZE
train_batch_size = 1
val_patch_size = TRAIN_PATCH_SIZE
val_batch_size = 1
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
    "scale": (0.75, 1.25),  # default
    "p_scale_per_sample": 0.1,
    # rotation
    "do_rotation": True,
    **{
        f"angle_{ax}": (-geom.deg_to_radians(15), geom.deg_to_radians(15))
        for ax in list("xyz")
    },
    "p_rot_per_sample": 0.1,
}
train_transform = SpatialTransform_2(**train_transform_args)


# NOTE this is not necessary to run the experiments and only used for wandb
# reporting
experiment_config = {
    "lr": lr,
    "betas": betas,
    "milestones": milestones,
    "lr_gamma": lr_gamma,
    "HULoss_args": HULoss_args,
    # ------------------------
    "generator": object_name(generator),
    "generator_optim": object_name(generator_optim),
    "generator_lr_scheduler": object_name(generator_lr_scheduler),
    "generator_args": generator_args,
    # ------------------------
    "discriminator": object_name(discriminator),
    "discriminator_optim": object_name(discriminator_optim),
    "discriminator_lr_scheduler": object_name(discriminator_lr_scheduler),
    "discriminator_args": discriminator_args,
    # ------------------------
    "train_patch_size": train_patch_size,
    "train_batch_size": train_batch_size,
    "val_patch_size": val_patch_size,
    "val_batch_size": val_batch_size,
    "dataset_paths": dataset_paths,
    "train_transform": object_name(train_transform),
    "train_transform_args": train_transform_args,
    # ------------------------
    "train_iterations": train_iterations,
    "train_generator_every": train_generator_every,
    "seed": seed,
    "fold_idx": fold_idx,
    "checkpoint_every": checkpoint_every,
}