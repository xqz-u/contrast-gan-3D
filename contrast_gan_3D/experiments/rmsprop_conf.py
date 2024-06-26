from functools import partial

from torch.optim import RMSprop

from contrast_gan_3D.experiments.basic_conf import lr
from contrast_gan_3D.experiments.small_patch_size import *

generator_optim_class = partial(RMSprop, lr=lr)
critic_optim_class = partial(RMSprop, lr=lr)
