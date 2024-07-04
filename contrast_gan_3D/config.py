from pathlib import Path

PROJECT_DIR = Path("./").resolve()

LOGS_DIR = PROJECT_DIR / "logs"
CHECKPOINTS_DIR = LOGS_DIR / "model_checkpoints"

DEFAULT_CVAL_SPLITS_PATH = PROJECT_DIR / "cross_val_splits.pkl"

ASSETS_DIR = PROJECT_DIR / "assets"

DATA_DIR = Path.home().resolve() / "data"
PREPROC_DATA_DIR = DATA_DIR / "preproc"
CENTERLINES_DIR = DATA_DIR / "auto_centerlines"
CORRECTIONS_DIR = DATA_DIR / "corrections"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
