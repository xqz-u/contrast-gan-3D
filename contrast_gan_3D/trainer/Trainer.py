from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from torch import Tensor, nn, profiler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import trange
from tqdm.contrib.itertools import product as tqdm_product

from contrast_gan_3D.alias import BGenAugmenter, ScanType
from contrast_gan_3D.config import CHECKPOINTS_DIR
from contrast_gan_3D.model.loss import WassersteinLoss, ZNCCLoss
from contrast_gan_3D.trainer.logger.LoggerInterface import (
    MultiThreadedLogger,
    SingleThreadedLogger,
)
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)

# TODO probably wrong validation metrics

# TODO gradient penalized Wasserstein loss instead of clipping network
#      parameters https://arxiv.org/pdf/1704.00028.pdf

# TODO use fold index to group cval runs belonging to same experiment together
# TODO save folds configuration so that restarting a wandb.run is guaranteed to
#      use the same train data as before it was stopped / save it into wandb itself!

# TODO `margin` parameter of batchgenerator's crop?
# TODO create few 3D scans of fake images to import into ITK-SNAP

# TODO AMP, DDP (?)


class Trainer:
    def __init__(
        self,
        train_iterations: int,
        val_iterations: int,
        validate_every: int,
        train_generator_every: int,
        log_every: int,
        log_images_every: int,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optim: Optimizer,
        discriminator_optim: Optimizer,
        hu_loss_instance: nn.Module,
        logger_interface: Union[SingleThreadedLogger, MultiThreadedLogger],
        run_id: str,
        device: torch.device,
        generator_lr_scheduler: Optional[LRScheduler] = None,
        discriminator_lr_scheduler: Optional[LRScheduler] = None,
        hu_loss_weight: float = 1.0,
        sim_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        checkpoint_every: Optional[int] = 1000,
    ):
        self.device = device
        logger.info("Using device: %s", self.device)

        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.val_every = validate_every
        self.train_generator_every = train_generator_every
        self.log_every = log_every
        self.log_images_every = log_images_every

        self.hu_loss_w = hu_loss_weight
        self.sim_loss_w = sim_loss_weight
        self.gan_loss_w = gan_loss_weight

        self.generator = generator
        self.optimizer_G = generator_optim
        self.lr_scheduler_G = generator_lr_scheduler

        self.discriminator = discriminator
        self.optimizer_D = discriminator_optim
        self.lr_scheduler_D = discriminator_lr_scheduler

        self.loss_GAN = WassersteinLoss()
        self.loss_similarity = ZNCCLoss()
        self.loss_HU = hu_loss_instance
        self.logger_interface = logger_interface

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = CHECKPOINTS_DIR / f"{run_id}.pt"
        self.checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        self.iteration = 0
        self.load_checkpoint(self.checkpoint_path)

    def train_step(self, patches: List[dict], iteration: int):
        self.optimizer_D.zero_grad(set_to_none=True)

        opt, low, high = patches
        opt = opt["data"].to(self.device, non_blocking=True)
        subopt = torch.cat([low["data"], high["data"]])
        subopt = subopt.to(self.device, non_blocking=True)

        # generate optimal image
        attenuation = self.generator(subopt)
        opt_hat: Tensor = subopt - attenuation

        # ------------------ discriminator
        D_x: Tensor = self.discriminator(opt)
        D_G_x: Tensor = self.discriminator(opt_hat.detach())

        loss_D: Tensor = self.gan_loss_w * self.loss_GAN(D_G_x, D_x)
        loss_D.backward()
        self.optimizer_D.step()
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        log_dict = {"D": loss_D, "D-real-out": D_x, "D-fake-out": D_G_x}

        # ------------------ generator
        if iteration % self.train_generator_every == 0:
            self.optimizer_G.zero_grad(set_to_none=True)
            subopt_mask = torch.cat([low["seg"], high["seg"]])
            subopt_mask = subopt_mask.to(self.device, non_blocking=True)

            loss_G = self.gan_loss_w * -self.loss_GAN(self.discriminator(opt_hat))
            loss_sim = self.sim_loss_w * self.loss_similarity(opt_hat, subopt)
            loss_hu = self.hu_loss_w * self.loss_HU(opt_hat, subopt_mask)
            if torch.isnan(loss_hu):
                loss_hu = torch.zeros(1, device=self.device)

            # full generator loss
            full_loss_G: Tensor = loss_G + loss_sim + loss_hu
            full_loss_G.backward()
            self.optimizer_G.step()

            log_dict.update(
                {"G": loss_G, "G-full": full_loss_G, "sim": loss_sim, "HU": loss_hu}
            )

        for tag in list("GD"):  # update learning rate schedulers, if any
            if (scheduler := getattr(self, f"lr_scheduler_{tag}")) is not None:
                scheduler.step()

        # ------------------ logging
        if iteration % self.log_every == 0:
            self.logger_interface.logger.log_loss(
                {k: v.mean() for k, v in log_dict.items()}, iteration, "train"
            )

        if iteration % self.log_images_every == 0:
            # method = self.wandb_logger.multiple
            method = self.logger_interface
            method(
                patches,
                [None, *opt_hat.chunk(2)],
                [None, *attenuation.chunk(2)],
                iteration,
                "train",
            )

    def fit(
        self,
        train_loaders: Dict[int, BGenAugmenter],
        val_loaders: Dict[int, BGenAugmenter],
        profiler: Optional[profiler.profile] = None,
    ):
        self.generator.train()
        self.discriminator.train()
        # start batchgenerator's async augmenters
        self.retrieve_batch_size(train_loaders[ScanType.OPT.value], "train")
        self._manage_augmenters([train_loaders, val_loaders], "start")

        for iteration in trange(self.iteration, self.train_iterations, desc="Train"):
            # NOTE order is determined by ScanType
            patches = [next(train_loaders[scan_type.value]) for scan_type in ScanType]
            self.train_step(patches, iteration)

            if iteration != 0 and iteration % self.val_every == 0:
                self.validate(val_loaders, iteration)

            if (
                self.checkpoint_every is not None
                and iteration % self.checkpoint_every == 0
            ):
                self.save_checkpoint(self.checkpoint_path, iteration)

            if profiler:
                profiler.step()

        if profiler:
            profiler.stop()
        self.save_checkpoint(self.checkpoint_path, self.train_iterations)
        self._manage_augmenters([train_loaders, val_loaders], "end")
        self.logger_interface.end_hook()

    def validate(self, val_loaders: Dict[int, BGenAugmenter], train_iteration: int):
        self.discriminator.eval()
        self.generator.eval()
        self.retrieve_batch_size(val_loaders[ScanType.OPT.value], "val")

        loss_sim, loss_G, loss_real_D, loss_fake_D = torch.zeros(
            4, dtype=torch.float32, device=self.device
        )
        loggable = []

        with torch.no_grad():
            for i, scan_type_enum in tqdm_product(
                range(self.val_iterations),
                ScanType,
                desc=f"Val {train_iteration // self.val_every}",
            ):
                batch = next(val_loaders[scan_type_enum.value])
                sample = batch["data"].to(self.device, non_blocking=True)
                sample_hat, attenuation = None, None

                if scan_type_enum == ScanType.OPT:
                    loss_real_D -= self.loss_GAN(self.discriminator(sample))
                else:
                    attenuation = self.generator(sample)
                    sample_hat = sample - attenuation
                    batch_loss_G = self.loss_GAN(self.discriminator(sample_hat))
                    loss_fake_D += batch_loss_G
                    loss_G -= batch_loss_G
                    loss_sim += self.loss_similarity(sample_hat, sample)

                if i == 0:
                    loggable.append([batch, sample_hat, attenuation])
                    if len(loggable) == len(ScanType):
                        patches, reconstructions, attenuations = list(zip(*loggable))
                        # method = self.wandb_logger.multiple
                        method = self.logger_interface
                        method(
                            patches,
                            list(reconstructions),
                            list(attenuations),
                            train_iteration,
                            "validation",
                        )
                        # eliminate references to free up GPU memory
                        for s, r, a in loggable:
                            del s, r, a
                        loggable = None

        self.discriminator.train()
        self.generator.train()

        # assumes dataloaders with same batch size
        n_opt = self.val_iterations * self.val_bs
        n_subopt = 2 * n_opt

        val_loss = {
            "D": (loss_real_D + loss_fake_D).mean(),
            "D-real": loss_real_D / n_opt,
        } | {
            k: v / n_subopt
            for k, v in zip(
                ["sim", "D-fake", "G"],
                [loss_sim, loss_fake_D, loss_G],
            )
        }
        self.logger_interface.logger.log_loss(val_loss, train_iteration, "validation")

    @property
    def model_torch_attrs(self) -> List[str]:
        return [
            "generator",
            "optimizer_G",
            "lr_scheduler_G",
            "discriminator",
            "optimizer_D",
            "lr_scheduler_D",
        ]

    def save_checkpoint(self, ckpt_path: Path, iteration: int):
        state = {"iteration": iteration}
        for attr in self.model_torch_attrs:
            el = getattr(self, attr, None)
            state[attr] = el if el is None else el.state_dict()
        torch.save(state, ckpt_path)

    def load_checkpoint(self, ckpt_path: Union[Path, str]):
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_file():
            logger.info("Resuming run from '%s'", str(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            for k, v in checkpoint.items():
                if k in self.model_torch_attrs:
                    if v is not None:
                        getattr(self, k).load_state_dict(v)
                else:
                    setattr(self, k, v)
        logger.info("Starting from iteration %d", self.iteration)

    @staticmethod
    def _manage_augmenters(augmenters: List[Dict[int, BGenAugmenter]], event: str):
        assert event in ["start", "end"], f"Unknown event {event!r}"
        for aug_dict in augmenters:
            for augmenter in aug_dict.values():
                if isinstance(
                    augmenter, (MultiThreadedAugmenter, NonDetMultiThreadedAugmenter)
                ):
                    if event == "start":
                        augmenter.restart()
                    else:
                        augmenter._finish()

    # NOTE only val_bs used atm :/
    def retrieve_batch_size(self, augmenter: BGenAugmenter, stage: str):
        attr_bs = f"{stage}_bs"
        if getattr(self, attr_bs, None) is None:
            if isinstance(
                augmenter, (MultiThreadedAugmenter, NonDetMultiThreadedAugmenter)
            ):
                aug_attr_name = "generator"
            else:
                aug_attr_name = "data_loader"
            setattr(self, attr_bs, getattr(augmenter, aug_attr_name).batch_size)
            setattr(self, f"{stage}_it_patches", len(ScanType) * getattr(self, attr_bs))
            setattr(
                self,
                f"{stage}_tot_patches",
                getattr(self, f"{stage}_iterations")
                * getattr(self, f"{stage}_it_patches"),
            )
