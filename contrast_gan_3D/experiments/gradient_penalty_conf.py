from torch import nn

from contrast_gan_3D.experiments.basic_conf import *
# from contrast_gan_3D.experiments.test_conf import *

weight_clip = None

# values from GP-WGAN paper, optimizer is Adam
betas = (0, 0.9)
lr = 1e-4
gp_weight = 10

# NOTE can't use LayerNorm as suggested in GP-WGAN paper when train and
# validation patch sizes differ
critic_args.update(
    {
        "norm_layer": nn.Identity,
        # "norm_layer": partial(nn.InstanceNorm3d, affine=True)
    }
)
critic_class = partial(PatchGANDiscriminator, **critic_args)
critic_optim_class = partial(Adam, lr=lr, betas=betas)
critic_lr_scheduler_class = partial(MultiStepLR, milestones=milestones, gamma=lr_gamma)
