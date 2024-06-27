from pathlib import Path

ROOT_DIR = Path("./").resolve()

LOGS_DIR = ROOT_DIR / "logs"

CHECKPOINTS_DIR = LOGS_DIR / "model_checkpoints"

DEFAULT_CVAL_SPLITS_PATH = ROOT_DIR / "cross_val_splits.pkl"

ASSETS_DIR = ROOT_DIR / "assets"
