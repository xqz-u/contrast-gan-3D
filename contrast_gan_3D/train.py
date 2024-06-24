import os

# https://discuss.pytorch.org/t/gpu-device-ordering/60785/2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Optional

import matplotlib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from wandb.sdk.lib.runid import generate_id

import wandb
from contrast_gan_3D import utils
from contrast_gan_3D.alias import FoldType
from contrast_gan_3D.config import CHECKPOINTS_DIR, LOGS_DIR, ROOT_DIR
from contrast_gan_3D.experiments.basic_conf import *
from contrast_gan_3D.model.loss import HULoss
from contrast_gan_3D.model.utils import count_parameters
from contrast_gan_3D.trainer import utils as train_u
from contrast_gan_3D.trainer.Trainer import Trainer
from contrast_gan_3D.utils.logging_utils import create_logger

matplotlib.use("agg")  # avoid MatplotLib warning about figures in threads

make_timestamp = lambda: time.strftime("%m_%d_%Y_%H_%M_%S")

logger = create_logger(name=__name__)


def maybe_create_profiler(
    profiler_dir: Optional[Path], device: torch.device
) -> Optional[torch.profiler.profile]:
    if profiler_dir is None:
        return
    # hard-set, global keyword necessary for amending
    global train_iterations, val_iterations, validate_every, checkpoint_every, log_every, log_images_every
    train_iterations, val_iterations = 61, 3
    validate_every, checkpoint_every = 10, None
    log_every, log_images_every = 10, 15

    logger.info("PyTorch profiler with TensorBoard trace in: '%s'", str(profiler_dir))
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


@dataclass
class TrainManager:
    wandb_project: str
    wandb_entity: str
    train_folds: list[FoldType]
    val_folds: list[FoldType]
    conf_overwrites: Optional[Path] = None
    wandb_run_id: Optional[str] = None
    device_idx: Optional[int] = None
    profiler_dir: Optional[Path] = None
    device: torch.device = field(init=False, default=None)
    profiler: Optional[torch.profiler.profile] = field(init=False, default=None)
    start_fold: int = field(init=False, default=0)
    group: str = field(
        init=False, default_factory=lambda: f"cval_experiment_{make_timestamp()}"
    )
    has_restarted: bool = field(init=False, default=False)

    def __post_init__(self):
        self.maybe_update_globals()
        self.ensure_reproducible()
        if self.wandb_run_id is not None:
            # restart interrupted experiment
            assert self.wandb_entity is not None, "Missing wandb entity."
            run_path = [self.wandb_entity, self.wandb_project, self.wandb_run_id]
            run = wandb.Api().run("/".join(run_path))
            self.group = run.group
            self.start_fold = run.config["fold"]
            logger.info(
                "RESUME run '%s' experiment '%s' fold %d",
                self.wandb_run_id,
                self.group,
                self.start_fold,
            )
        # setup other attributes
        self.device = utils.set_GPU(self.device_idx)
        self.profiler = maybe_create_profiler(self.profiler_dir, self.device)

    def maybe_update_globals(self):
        if self.conf_overwrites is not None:
            if self.wandb_run_id is not None:
                # NOTE when restarting a run, check the right overwrites are being
                # used from the stopped run's wandb Overview page, section 'Command'
                logger.warning(
                    "Restart run '%s' reading overwrites from file", self.wandb_run_id
                )
            logger.info("Reading overwrites from '%s'", str(self.conf_overwrites))
            overwrite_module = train_u.global_overrides(self.conf_overwrites)
            globals().update(vars(overwrite_module))

    def ensure_reproducible(self):
        if seed is not None:
            logger.info("Using seed %d", seed)
            utils.seed_everything(seed)
        else:
            # NOTE increase speed but halts reproducibility
            logger.info("Set CUDNN in benchmark mode")
            torch.backends.cudnn.benchmark = True

    def generate_run_id(self) -> str:
        if self.wandb_run_id is not None and not self.has_restarted:
            self.has_restarted = True
            return self.wandb_run_id
        return generate_id()  # new id for each cross validation run

    def __call__(self):
        for fold, (train_fold, val_fold) in enumerate(
            zip(self.train_folds[self.start_fold :], self.val_folds[self.start_fold :]),
            start=self.start_fold,
        ):
            run_id = self.generate_run_id()

            train_loaders, val_loaders = train_u.create_dataloaders(
                train_fold,
                val_fold,
                train_patch_size,
                val_patch_size,
                train_batch_size,
                val_batch_size,
                rng,
                scaler=scaler,
                num_workers=num_workers,
                train_transform=train_transform,
                seed=seed,
            )

            scaled_HU_bounds = scaler(np.array(desired_HU_bounds))
            logger.info(
                "Desired HU bounds: %s scaled: %s", desired_HU_bounds, scaled_HU_bounds
            )
            train_subopt_bs = (
                train_batch_size[ScanType.LOW.value]
                + train_batch_size[ScanType.HIGH.value]
            )
            trainer = Trainer(
                train_iterations,
                val_iterations,
                validate_every,
                train_generator_every,
                train_critic_every,
                log_every,
                log_images_every,
                generator_class,
                critic_class,
                generator_optim_class,
                critic_optim_class,
                HULoss(*scaled_HU_bounds, (train_subopt_bs, 1, *train_patch_size)),
                logger_interface,
                checkpoint_dir=CHECKPOINTS_DIR / run_id,
                weight_clip=weight_clip,
                generator_lr_scheduler_class=critic_lr_scheduler_class,
                critic_lr_scheduler_class=critic_lr_scheduler_class,
                device=self.device,
                checkpoint_every=checkpoint_every,
                rng=rng,
            )

            critic_size = count_parameters(trainer.critic)
            generator_size = count_parameters(trainer.generator)
            logger.info(
                "Critic size: %d Generator size: %d", critic_size, generator_size
            )

            local_conf = {
                "generator": trainer.generator,
                "critic": trainer.critic,
                "fold": fold,
                "generator_size": generator_size,
                "critic_size": critic_size,
            }
            experiment_config = local_conf | train_u.config_from_globals(globals())
            pprint(experiment_config)

            logger.info("FOLD %d", fold)
            with wandb.init(
                id=run_id,
                resume="allow",
                project=self.wandb_project,
                entity=self.wandb_entity,
                dir=LOGS_DIR,
                config=experiment_config,
                group=self.group,
            ) as run:
                logger_interface.logger.setup_wandb_run(run)
                trainer.fit(train_loaders, val_loaders, profiler=self.profiler)
                if self.profiler is not None:  # profile only one run
                    break

            break  # just do one fold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--conf-overwrites",
        type=Path,
        help="Optional path to a .py file defining experiment variables overwrites.",
        default=None,
    )
    parser.add_argument("--wandb-project", type=str, default="contrast-gan-3D")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="wandb run id used to resume training runs.",
    )
    parser.add_argument(
        "--profiler-dir", type=Path, default=None, help="torch-tb-profiler logs dir."
    )
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument(
        "--cross-validation-splits",
        default=ROOT_DIR / "cross_val_splits.pkl",
        type=Path,
        help="Path to Pickle file defining train/test splits.",
    )
    args = parser.parse_args()

    train_val_file = args.cross_validation_splits
    assert train_val_file.is_file(), "Run notebooks/create_dataset.ipynb first."
    logger.info("Reading train/test splits from '%s'", str(train_val_file))
    cval = load_pickle(train_val_file)

    TrainManager(
        args.wandb_project,
        args.wandb_entity,
        cval["train"],
        cval["test"],
        args.conf_overwrites,
        args.wandb_run_id,
        args.device,
        args.profiler_dir,
    )()
