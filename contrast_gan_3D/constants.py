import numpy as np

AORTIC_ROOT_PATCH_SIZE = np.array([19] * 3)
AORTIC_ROOT_PATCH_SPACING = np.array([0.5] * 3)

# level: 240, window: 1000
VMIN, VMAX = -260, 740

ORIENTATION = "LPS"

TRAIN_PATCH_SIZE = np.array((128,) * 3)
DEFAULT_SEED = 42
