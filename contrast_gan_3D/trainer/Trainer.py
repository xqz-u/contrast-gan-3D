from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import make_grid
from tqdm.auto import trange

import wandb
from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.config import CHECKPOINTS_DIR
from contrast_gan_3D.model.loss import HULoss, WassersteinLoss, ZNCCLoss
from contrast_gan_3D.trainer.Reloader import Reloader
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


# TODO clip logged images to right window --- dynamic
# TODO models DataParallel
# TODO inference: patches aggregation
class Trainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optim: Optimizer,
        discriminator_optim: Optimizer,
        train_generator_every: int,
        run_id: str,
        device: torch.device,
        generator_lr_scheduler: Optional[LRScheduler] = None,
        discriminator_lr_scheduler: Optional[LRScheduler] = None,
        hu_loss_weight: float = 1.0,
        sim_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        checkpoint_every: int = 1000,
        rng: Optional[np.random.Generator] = None,
        **hu_loss_kwargs,
    ):
        self.device = device
        self.rng = rng or np.random.default_rng()

        self.hu_loss_w = hu_loss_weight
        self.sim_loss_w = sim_loss_weight
        self.gan_loss_w = gan_loss_weight

        self.train_generator_every = train_generator_every

        self.generator = generator
        self.optimizer_G = generator_optim
        self.lr_scheduler_G = generator_lr_scheduler

        self.discriminator = discriminator
        self.optimizer_D = discriminator_optim
        self.lr_scheduler_D = discriminator_lr_scheduler

        self.loss_GAN = WassersteinLoss()
        self.loss_similarity = ZNCCLoss()
        self.loss_HU = HULoss(**hu_loss_kwargs)

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = CHECKPOINTS_DIR / f"{run_id}.pt"
        self.checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        self.iteration = 0
        self.load_checkpoint()

    def train_step(
        self,
        opt: torch.Tensor,
        subopt: torch.Tensor,
        subopt_mask: torch.Tensor,
        iteration: int,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        self.optimizer_D.zero_grad()
        # generate optimal image
        opt_hat = subopt - self.generator(subopt)
        # discriminator
        loss_D = self.gan_loss_w * self.loss_GAN(
            self.discriminator(opt_hat.detach()), self.discriminator(opt)
        )
        loss_D.backward()
        self.optimizer_D.step()
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        ret = {"D": loss_D.cpu().item()}
        # generator
        if iteration % self.train_generator_every == 0:
            self.optimizer_G.zero_grad()
            # generator's individual loss components
            loss_a = -self.loss_GAN(self.discriminator(opt_hat))
            loss_i = self.loss_similarity(opt_hat, subopt)
            loss_hu = self.loss_HU(opt_hat, subopt_mask)
            # full generator loss
            loss_G = self.gan_loss_w * loss_a + self.sim_loss_w * loss_i
            if not torch.isnan(loss_hu):
                loss_G += self.hu_loss_w * loss_hu
            loss_G.backward()
            self.optimizer_G.step()

            for k, v in zip(
                ["G", "adversarial", "similarity", "HU"],
                [loss_G, loss_a, loss_i, loss_hu],
            ):
                ret[k] = v.cpu().item()

        # update learning rate schedulers, if any
        for tag in list("GD"):
            if (scheduler := getattr(self, f"lr_scheduler_{tag}")) is not None:
                scheduler.step()

        return ret, opt_hat

    def fit(
        self,
        train_iterations: int,
        validate_every: int,
        train_loaders: Dict[int, Reloader],
        val_loaders: Dict[int, Reloader],
    ):
        self.generator.to(self.device).train()
        self.discriminator.to(self.device).train()

        logger.info("Using device: %s", self.device)
        last_G_loss = None

        for iteration in (pbar := trange(self.iteration, train_iterations)):
            pbar.set_description(f"Train iteration {iteration}")
            opt = next(train_loaders[ScanType.OPT.value])
            low = next(train_loaders[ScanType.LOW.value])
            high = next(train_loaders[ScanType.HIGH.value])
            # to GPU
            for el in [opt, low, high]:
                for k in ["data", "seg"]:
                    el[k] = el[k].to(self.device)

            subopt = torch.cat([low["data"], high["data"]])
            subopt_mask = torch.cat([low["seg"], high["seg"]])

            train_loss, opt_hat = self.train_step(
                opt["data"], subopt, subopt_mask, iteration
            )

            if "G" in train_loss:
                last_G_loss = train_loss["G"]
            pbar.set_postfix(**{"D": train_loss["D"], "G": last_G_loss})

            self.log_loss(train_loss, iteration, "train")

            if iteration % validate_every == 0:
                self.log_loss(
                    self.validate(val_loaders, iteration), iteration, "validation"
                )
                # reconstruction and low/high should be logged with same indices
                self.log_images(opt["data"], iteration, "train", ScanType.OPT.name)
                self.log_images(
                    low["data"],
                    iteration,
                    "train",
                    ScanType.LOW.name,
                    reconstructions=opt_hat[: len(low["data"])],
                )
                self.log_images(
                    high["data"],
                    iteration,
                    "train",
                    ScanType.HIGH.name,
                    reconstructions=opt_hat[len(low["data"]) :],
                )

            if iteration % self.checkpoint_every == 0:
                self.save_checkpoint(iteration)
        # final checkpoint
        self.save_checkpoint(train_iterations)

    def validate(
        self, val_loaders: Dict[int, Reloader], iteration: int
    ) -> Dict[str, float]:
        self.discriminator.eval().to(self.device)
        self.generator.eval().to(self.device)

        loss_i, loss_a, loss_real, loss_fake = torch.zeros(
            4, dtype=torch.float, device=self.device
        )

        with torch.no_grad():
            for scan_type in ScanType:
                loader = val_loaders[scan_type.value]
                for i, batch in enumerate(loader):
                    sample = batch["data"].to(self.device)
                    if scan_type == ScanType.OPT:
                        loss_real -= self.loss_GAN(self.discriminator(sample))
                        if i == 0:
                            self.log_images(
                                sample, iteration, "validation", scan_type.name
                            )
                    else:
                        sample_hat = sample - self.generator(sample)
                        adversarial_loss = self.loss_GAN(self.discriminator(sample_hat))
                        loss_fake += adversarial_loss
                        loss_a -= adversarial_loss
                        loss_i += self.loss_similarity(sample_hat, sample)
                        if i == 0:
                            self.log_images(
                                sample,
                                iteration,
                                "validation",
                                scan_type.name,
                                reconstructions=sample_hat,
                            )

        n_opt = len(val_loaders[ScanType.OPT.value].dataset)
        n_low = len(val_loaders[ScanType.LOW.value].dataset)
        n_high = len(val_loaders[ScanType.HIGH.value].dataset)
        n_subopt = n_low + n_high

        ret = {
            "D": (loss_real + loss_fake).mean(),
            "adversarial_real": loss_real / n_opt,
        }
        for k, v in zip(
            ["similarity", "adversarial_fake", "adversarial"],
            [loss_i, loss_fake, loss_a],
        ):
            ret[k] = v.cpu().item() / n_subopt

        self.discriminator = self.discriminator.train()
        self.generator = self.generator.train()

        return ret

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

    def save_checkpoint(self, iteration: int):
        state = {"iteration": iteration}
        for attr in self.model_torch_attrs:
            el = getattr(self, attr, None)
            state[attr] = el if el is None else el.state_dict()
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self):
        if self.checkpoint_path.is_file():
            logger.info("Resuming run from '%s'", str(self.checkpoint_path))
            checkpoint = torch.load(
                self.checkpoint_path, map_location=torch.device("cpu")
            )
            for k, v in checkpoint.items():
                if k in self.model_torch_attrs:
                    if v is not None:
                        getattr(self, k).load_state_dict(v)
                else:
                    setattr(self, k, v)
        logger.info("Starting from iteration %d", self.iteration)

    @staticmethod
    def log_loss(loss_dict: Dict[str, float], iteration: int, stage: str):
        wandb.log({f"{stage}/{k}": v for k, v in loss_dict.items()}, step=iteration)

    # do 12 slices 3x4 of a random sample in a batch
    def log_images(
        self,
        batch: torch.Tensor,
        iteration: int,
        stage: str,
        tag: str,
        reconstructions: Optional[torch.Tensor] = None,
    ):
        sample_idx = self.rng.integers(len(batch))
        sample = batch[sample_idx]
        slice_idxs = sorted(self.rng.choice(sample.shape[-1], size=12, replace=False))
        slices = sample.permute(3, 0, 2, 1)[slice_idxs]  # shapes: CWHD -> DCHW
        images = wandb.Image(make_grid(slices, nrow=4))
        wandb.log({f"{stage}/images/{tag}": images}, step=iteration)

        if reconstructions is not None:
            slices = reconstructions[sample_idx].permute(3, 0, 2, 1)[slice_idxs]
            images = wandb.Image(make_grid(slices, nrow=4))
            wandb.log({f"{stage}/images/{tag}_recon": images}, step=iteration)
