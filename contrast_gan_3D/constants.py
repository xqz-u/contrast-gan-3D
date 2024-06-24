import numpy as np

AORTIC_ROOT_PATCH_SIZE = np.array([19] * 3)
AORTIC_ROOT_PATCH_SPACING = np.array([0.5] * 3)

MIN_HU, MAX_HU = -1024, 1500
VMIN, VMAX = -260, 740  # level: 240, window: 1000 for display

ORIENTATION = "LPS"

TRAIN_PATCH_SIZE = (128,) * 3
VAL_PATCH_SIZE = (256, 256, 128)
DEFAULT_SEED = 42
