from contrast_gan_3D.experiments.basic_conf import *

# train_iterations = 12
# val_iterations = 5
# checkpoint_every = None
# validate_every = 4
# log_every = 2
# log_images_every = 3

train_iterations, val_iterations = 61, 3
validate_every, checkpoint_every = 10, None
log_every, log_images_every = 10, 15

train_batch_size = 3  # 9 tot
val_batch_size = 2  # 6 tot
num_workers = (train_batch_size * 2, val_batch_size * 2)  # (train, validation)
