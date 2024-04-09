import os

# https://discuss.pytorch.org/t/gpu-device-ordering/60785/2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from pathlib import Path
from pprint import pprint
from typing import Optional

import matplotlib
import numpy as np
import torch
from wandb.sdk.lib.runid import generate_id

import wandb
from contrast_gan_3D import utils
from contrast_gan_3D.config import CHECKPOINTS_DIR, LOGS_DIR
from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.model.loss import HULoss
from contrast_gan_3D.model.utils import count_parameters
from contrast_gan_3D.trainer import utils as train_utils
from contrast_gan_3D.trainer.Trainer import Trainer
from contrast_gan_3D.utils.logging_utils import create_logger

matplotlib.use("agg")  # avoid MatplotLib warning about figures in threads

logger = create_logger(name=__name__)


def create_profiler(profiler_dir: Path, device: torch.device) -> torch.profiler.profile:
    # hard-set
    global train_iterations, val_iterations, validate_every, checkpoint_every, log_every, log_images_every
    train_iterations, val_iterations = 61, 3
    validate_every, checkpoint_every = 10, None
    log_every, log_images_every = 10, 15

    logger.info("PyTorch profiler with TensorBoard trace in: '%s'", profiler_dir)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if "cuda" in device.type:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(skip_first=11, wait=3, warmup=4, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def main(
    wandb_project: str,
    wandb_entity: str,
    device_idx: Optional[int],
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
            logger.info("NEW run_id: '%s'", run_id)
        else:
            logger.info("OLD run_id: '%s'", run_id)

        device = utils.set_GPU(device_idx)

        profiler = None
        if profiler_dir is not None:
            profiler = create_profiler(Path(f"{str(profiler_dir)}_{i}"), device)

        trainer = Trainer(
            train_iterations,
            val_iterations,
            validate_every,
            train_generator_every,
            log_every,
            log_images_every,
            generator_class,
            critic_class,
            generator_optim_class,
            critic_optim_class,
            # train_bantch_size * 2 == [low batch, high batch]
            HULoss(*scaled_HU_bounds, (train_batch_size * 2, 1, *train_patch_size)),
            logger_interface,
            CHECKPOINTS_DIR / f"{run_id}.pt",
            weight_clip=weight_clip,
            generator_lr_scheduler_class=critic_lr_scheduler_class,
            critic_lr_scheduler_class=critic_lr_scheduler_class,
            device=device,
            checkpoint_every=checkpoint_every,
        )

        experiment_config = train_utils.update_experiment_config(globals()) | {
            "generator": trainer.generator,
            "critic": trainer.critic,
        }
        pprint(experiment_config)
        logger.info(
            "Critic size: %d Generator size: %d",
            count_parameters(trainer.critic),
            count_parameters(trainer.generator),
        )

        with wandb.init(
            id=run_id,
            resume="allow",
            project=wandb_project,
            entity=wandb_entity,
            dir=LOGS_DIR,
            config=experiment_config,
        ) as run:
            logger_interface.logger.setup_run(run)
            trainer.fit(train_loaders, val_loaders, profiler=profiler)
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
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="wandb run id used to resume logging a run.",
    )
    parser.add_argument(
        "--profiler-dir", type=Path, default=None, help="torch-tb-profiler logs dir."
    )
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    args = parser.parse_args()

    override_module = args.conf_overrides
    if override_module is not None:
        logger.info("Reading overrides from '%s'", str(override_module))
        override_module = train_utils.global_overrides(override_module)
        globals().update(vars(override_module))

    main(
        args.wandb_project,
        args.wandb_entity,
        args.device,
        run_id=args.wandb_run_id,
        profiler_dir=args.profiler_dir,
    )
