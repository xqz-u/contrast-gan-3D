from torch import nn

from contrast_gan_3D.experiments.basic_conf import *

# from contrast_gan_3D.experiments.test_conf import *

weight_clip = None
# values from GP-WGAN paper, optimizer is Adam
betas = (0, 0.9)
lr = 1e-4
gp_weight = 10

# NOTE if using LayerNorm, remember to skip validation
critic_args.update(norm_layer=nn.Identity)
critic_class = partial(PatchGANDiscriminator, **critic_args)
