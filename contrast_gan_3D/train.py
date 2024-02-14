import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from wandb.sdk.lib.runid import generate_id

import wandb
from contrast_gan_3D.config import LOGS_DIR
from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.trainer.Trainer import Trainer
from contrast_gan_3D.trainer.utils import create_train_folds

# TODO debug level argument on CLI


# author: ChatGPT
def overrides(config_path: Path):
    # Check if the path exists
    if not os.path.exists(config_path):
        print("Error: Config file does not exist.")
        return

    # Get the directory and file name from the path
    config_dir, config_file = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_file)

    # Add the directory to the sys.path if not already there
    if config_dir not in sys.path:
        sys.path.append(config_dir)

    # Use importlib to load the module
    return importlib.import_module(config_name)


def main(
    wandb_project: str,
    wandb_entity: str,
    run_id: Optional[str] = None,
):
    folds = create_train_folds(
        train_patch_size,
        val_patch_size,
        train_batch_size,
        val_batch_size,
        *dataset_paths,
        max_HU_diff=max_HU_diff,
        train_transform=train_transform,
        seed=seed,
    )
    train_loaders, val_loaders = folds[fold_idx]

    if run_id is not None:
        print(f"Given run_id: {run_id!r}")
    else:
        run_id = generate_id()
        print(f"New run_id: {run_id!r}")

    wandb.init(
        id=run_id,
        resume="allow",
        project=wandb_project,
        entity=wandb_entity,
        dir=LOGS_DIR,
        config=experiment_config,
    )

    trainer = Trainer(
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        train_generator_every,
        run_id,
        generator_lr_scheduler=generator_lr_scheduler,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        device=device,
        checkpoint_every=checkpoint_every,
        rng=np.random.default_rng(seed),
        **HULoss_args,
    )
    trainer.fit(train_iterations, validate_every, train_loaders, val_loaders)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--conf-overrides",
        type=Path,
        help="Optional path to a .py file defining experiment variables overrides.",
        default=None,
    )
    parser.add_argument("--wandb-project", type=str, default="contrast-gan-3D")
    parser.add_argument("--wandb-entity", type=str, default="xqz-u")
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="wandb run id used to resume logging a run.",
    )
    args = parser.parse_args()

    override_module = args.conf_overrides
    if override_module is not None:
        print(f"Reading overrides from {str(override_module)!r}")
        override_module = overrides(override_module)
        globals().update(vars(override_module))

    main(args.wandb_project, args.wandb_entity, run_id=args.wandb_run_id)
