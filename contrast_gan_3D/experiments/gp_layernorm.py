from contrast_gan_3D.experiments.basic_conf import train_patch_size
from contrast_gan_3D.experiments.gradient_penalty_conf import *
from contrast_gan_3D.experiments.small_patch_size import train_patch_size

validate_every = None

num_workers = (3, 1)

critic_args.update(
    norm_layer=nn.LayerNorm, patch_size=(1, *train_patch_size), elementwise_affine=False
)
critic_class = partial(PatchGANDiscriminator, **critic_args)
