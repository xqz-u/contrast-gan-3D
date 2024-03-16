train_iterations = 10
train_generator_every = 2
checkpoint_every = 10
validate_every = 2
log_every = 1
log_images_every = 2

train_batch_size = 4  # 12 tot
val_batch_size = 2  # 6 tot
num_workers = (train_batch_size * 2, val_batch_size * 2)  # (train, validation)
