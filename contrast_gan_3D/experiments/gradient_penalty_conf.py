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
critic_args.update(norm_layer=nn.Identity)
critic_class = partial(PatchGANDiscriminator, **critic_args)

# NOTE default train_batch_size is fine on a GPU with ~20GB VRAM
train_batch_size = {
    v.value: b for v, b in [(ScanType.OPT, 6), (ScanType.LOW, 3), (ScanType.HIGH, 3)]
}
