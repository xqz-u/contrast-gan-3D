from contrast_gan_3D.data.Scaler import MinMaxScaler
from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.trainer.ImageLogger import MinMaxImageLogger

scaler = MinMaxScaler(*HU_norm_range)
image_logger = MinMaxImageLogger(scaler, rng=np.random.default_rng(seed=seed))
