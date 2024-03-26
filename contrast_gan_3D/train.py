import os

from contrast_gan_3D.data.Scaler import FactorMinMaxScaler, MinMaxScaler

# https://discuss.pytorch.org/t/gpu-device-ordering/60785/2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from wandb.sdk.lib.runid import generate_id

import wandb
from contrast_gan_3D import utils
from contrast_gan_3D.config import LOGS_DIR
from contrast_gan_3D.data.utils import compute_dataset_mean, minmax_norm
from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.model.loss import HULoss
from contrast_gan_3D.trainer import utils as train_utils
from contrast_gan_3D.trainer.Trainer import Trainer
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


def main(
    wandb_project: str,
    wandb_entity: str,
    run_id: Optional[str] = None,
    profiler_dir: Optional[Path] = None,
):
    if seed is not None:
        logger.info("Using seed %d", seed)
        utils.seed_everything(seed)
    else:
        # NOTE increase speed but halts reproducibility, turn off afterwards
        logger.info("Set CUDNN in benchmark mode")
        torch.backends.cudnn.benchmark = True

    train_folds, val_folds = train_utils.cval_paths(cval_folds, *dataset_paths)

    for i, (train_fold, val_fold) in enumerate(zip(train_folds, val_folds)):
        if isinstance(scaler, MinMaxScaler):
            logger.info("Computing train data mean for fold %d", i)
            train_mean = compute_dataset_mean(*[p for p, _ in train_fold])
            norm_train_mean = minmax_norm(train_mean, HU_norm_range)
            logger.info(
                "Train data mean: %.3f normalized %s: %.3f",
                train_mean,
                HU_norm_range,
                norm_train_mean,
            )
            scaler.b = norm_train_mean
            generator.scaler = FactorMinMaxScaler(
                *HU_norm_range, max_HU_delta, b=norm_train_mean
            )

        scaled_HU_bounds = scaler(np.array(desired_HU_bounds))
        logger.info(
            "Desired HU bounds: %s scaled: %s", desired_HU_bounds, scaled_HU_bounds
        )

        train_loaders, val_loaders = train_utils.create_dataloaders(
            train_fold,
            val_fold,
            train_patch_size,
            val_patch_size,
            train_batch_size,
            val_batch_size,
            scaler=scaler,
            num_workers=num_workers,
            train_transform=train_transform,
            seed=seed,
        )

        if run_id is None:
            run_id = generate_id()
            logger.info(f"NEW run_id: {run_id!r}")
        else:
            logger.info(f"OLD run_id: {run_id!r}")

        trainer = Trainer(
            train_iterations,
            val_iterations,
            validate_every,
            train_generator_every,
            log_every,
            log_images_every,
            val_batch_size,
            generator,
            discriminator,
            generator_optim,
            discriminator_optim,
            # train_bantch_size * 2 == [low batch, high batch]
            HULoss(*scaled_HU_bounds, (train_batch_size * 2, 1, *train_patch_size)),
            image_logger,
            run_id,
            generator_lr_scheduler=generator_lr_scheduler,
            discriminator_lr_scheduler=discriminator_lr_scheduler,
            device=device,
            checkpoint_every=checkpoint_every,
            profiler_dir=profiler_dir,
        )

        experiment_config = train_utils.update_experiment_config(globals())
        pprint(experiment_config)

        wandb.init(
            id=run_id,
            resume="allow",
            project=wandb_project,
            entity=wandb_entity,
            dir=LOGS_DIR,
            config=experiment_config,
        )

        trainer.fit(train_loaders, val_loaders)
        break


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
    parser.add_argument(
        "--profiler-dir", type=Path, default=None, help="torch-tb-profiler logs dir."
    )
    args = parser.parse_args()

    override_module = args.conf_overrides
    if override_module is not None:
        logger.info(f"Reading overrides from {str(override_module)!r}")
        override_module = train_utils.global_overrides(override_module)
        globals().update(vars(override_module))

    main(
        args.wandb_project,
        args.wandb_entity,
        run_id=args.wandb_run_id,
        profiler_dir=args.profiler_dir,
    )
