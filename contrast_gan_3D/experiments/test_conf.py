train_iterations = 10
val_iterations = 5
train_generator_every = 2
checkpoint_every = 10
validate_every = 5
log_every = 5
log_images_every = 5

train_batch_size = 4  # 12 tot
val_batch_size = 3  # 9 tot
num_workers = (train_batch_size * 2, val_batch_size * 2)  # (train, validation)
