from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.profiler
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import make_grid
from tqdm.auto import trange

import wandb
from contrast_gan_3D.alias import BGenAugmenter, ScanType
from contrast_gan_3D.config import CHECKPOINTS_DIR
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.model.loss import HULoss, WassersteinLoss, ZNCCLoss
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)

# TODO AMP?
# TODO other generator/discriminator convolutional architectures

# TODO plot offset masks during train/validation too (NOT OFTEN), look at Roel's appendix plots
# TODO do not log coronary arteries centerlines mask, they are mostly zeroed

# TODO models DDP (?)
# TODO inference: patches aggregation

# TODO try gradient penalized Wasserstein loss instead of clipping network
#      parameters

# TODO reproducibility

# TODO restore experiment configuration / run state fully from w&b when run is resumed
# TODO log images in background thread

# TODO keep profiler in optionally


class Trainer:
    def __init__(
        self,
        train_iterations: int,
        val_iterations: int,
        validate_every: int,
        train_generator_every: int,
        train_bs: int,
        val_bs: int,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optim: Optimizer,
        discriminator_optim: Optimizer,
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

        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.val_every = validate_every
        self.train_generator_every = train_generator_every

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
        self.loss_HU = HULoss(**hu_loss_kwargs)

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = CHECKPOINTS_DIR / f"{run_id}.pt"
        self.checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        self.iteration = 0
        self.load_checkpoint()

        self.train_bs = train_bs
        self.val_bs = val_bs

    def train_step(
        self,
        opt: Tensor,
        subopt: Tensor,
        subopt_mask: Tensor,
        subopt_min: Tensor,
        iteration: int,
    ) -> Tuple[Dict[str, float], Tensor]:
        self.optimizer_D.zero_grad(set_to_none=True)
        # generate optimal image
        opt_hat = subopt - self.generator(subopt)

        # ------------------ discriminator
        D_x = self.discriminator(opt)
        D_G_x = self.discriminator(opt_hat.detach())
        loss_D = self.gan_loss_w * self.loss_GAN(D_G_x, D_x)

        loss_D.backward()
        self.optimizer_D.step()
        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        info = {"D": loss_D, "D_x": D_x.mean(), "D_G_x": D_G_x.mean()}

        # ------------------ generator
        if iteration % self.train_generator_every == 0:
            self.optimizer_G.zero_grad(set_to_none=True)

            loss_G = self.gan_loss_w * -self.loss_GAN(self.discriminator(opt_hat))
            loss_sim = self.sim_loss_w * self.loss_similarity(opt_hat, subopt)
            loss_hu = self.loss_HU(opt_hat, subopt_mask, subopt_min)
            if not torch.isnan(loss_hu):
                loss_hu *= self.hu_loss_w
            # full generator loss
            full_loss_G = loss_G + loss_sim + loss_hu

            full_loss_G.backward()
            self.optimizer_G.step()
            info.update(
                zip(
                    ["G_full", "G", "similarity", "HU"],
                    [full_loss_G, loss_G, loss_sim, loss_hu],
                )
            )

        # update learning rate schedulers, if any
        for tag in list("GD"):
            if (scheduler := getattr(self, f"lr_scheduler_{tag}")) is not None:
                scheduler.step()

        return info, opt_hat

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

    def fit(
        self,
        train_loaders: Dict[int, BGenAugmenter],
        val_loaders: Dict[int, BGenAugmenter],
        log_every: int,
    ):
        logger.info("Using device: %s", self.device)

        self.generator.train()
        self.discriminator.train()

        self._manage_augmenters([train_loaders, val_loaders], "start")

        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(skip_first=5, wait=2, warmup=3, active=10),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #         LOGS_DIR / "profiler"
        #     ),
        #     profile_memory=True,
        #     with_stack=True,
        # ) as prof:
        with nullcontext():
            for iteration in trange(
                self.iteration, self.train_iterations, desc="Train", unit="batch"
            ):
                # if iteration + 1 >= 65:
                #     break
                # prof.step()

                opt, low, high = [
                    next(train_loaders[scan_type.value]) for scan_type in ScanType
                ]

                subopt = torch.cat([low["data"], high["data"]])
                subopt_mask = torch.cat([low["seg"], high["seg"]])
                mins = [el["meta"]["min"] for el in [opt, low, high]]
                subopt_min = torch.cat(mins[1:])
                # to GPU
                arrays = [
                    el.to(self.device, non_blocking=True)
                    for el in [opt["data"], subopt, subopt_mask, subopt_min]
                ]

                train_loss, opt_hat = self.train_step(*arrays, iteration)

                if iteration % log_every == 0:
                    self.log_loss(train_loss, iteration, "train")
                    # reconstruction and low/high should be logged with the same indices
                    n_low = len(low["data"])
                    for data, min_, scan_type, recon in zip(
                        [opt, low, high],
                        mins,
                        ScanType,
                        [None, opt_hat[:n_low], opt_hat[n_low:]],
                    ):
                        self.log_images(
                            data["data"],
                            min_,
                            data["seg"],
                            iteration,
                            "train",
                            scan_type.name,
                            reconstructions=recon,
                        )
                # cleanup for GPU memory consumption
                for el in train_loss.values():
                    del el
                del train_loss, opt_hat

                if iteration != 0 and iteration % self.val_every == 0:
                    val_loss = self.validate(val_loaders, iteration)
                    self.log_loss(val_loss, iteration, "validation")
                    for el in val_loss.values():
                        del el

                if iteration % self.checkpoint_every == 0:
                    self.save_checkpoint(iteration)
        # final checkpoint
        self.save_checkpoint(self.train_iterations)
        self._manage_augmenters([train_loaders, val_loaders], "end")

    def validate(
        self, val_loaders: Dict[int, BGenAugmenter], train_iteration: int
    ) -> Dict[str, float]:
        self.discriminator.eval()
        self.generator.eval()

        loss_sim, loss_G, loss_real_D, loss_fake_D = torch.zeros(
            4, dtype=torch.float, device=self.device
        )

        with torch.no_grad():
            for scan_type in ScanType:
                loader = val_loaders[scan_type.value]
                for i in trange(
                    self.val_iterations,
                    desc=f"Val {train_iteration // self.val_every} <{scan_type.name}>",
                    unit="batch",
                ):
                    batch = next(loader)
                    sample = batch["data"].to(self.device, non_blocking=True)
                    min_ = batch["meta"]["min"]
                    recon = None

                    if scan_type == ScanType.OPT:
                        loss_real_D -= self.loss_GAN(self.discriminator(sample))
                    else:
                        sample_hat = sample - self.generator(sample)
                        batch_loss_G = self.loss_GAN(self.discriminator(sample_hat))
                        loss_fake_D += batch_loss_G
                        loss_G -= batch_loss_G
                        loss_sim += self.loss_similarity(sample_hat, sample)
                        recon = sample_hat

                    if i == 0:
                        self.log_images(
                            sample,
                            min_,
                            batch["seg"],
                            train_iteration,
                            "validation",
                            scan_type.name,
                            reconstructions=recon,
                        )

        self.discriminator.train()
        self.generator.train()

        n_opt = self.val_iterations * self.val_bs
        n_subopt = 2 * n_opt

        return {
            "D": (loss_real_D + loss_fake_D).mean(),
            "D_real": loss_real_D / n_opt,
        } | {
            k: v / n_subopt
            for k, v in zip(
                ["similarity", "D_fake", "G"],
                [loss_sim, loss_fake_D, loss_G],
            )
        }

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
    def log_loss(loss_dict: Dict[str, Tensor], iteration: int, stage: str):
        wandb.log({f"{stage}/{k}": v for k, v in loss_dict.items()}, step=iteration)

    def _log_images_grid(
        self,
        sample: Tensor,
        slice_idxs: Tensor,
        tag: str,
        iteration: int,
        sample_min: Optional[Tensor] = None,
    ) -> Tensor:
        # CWHD -> DCHW -> CHW for `make_grid`
        slices = sample.permute(3, 0, 2, 1)[slice_idxs]
        if sample_min is not None:
            slices = torch.clip(slices * self.loss_HU.HU_diff - sample_min, VMIN, VMAX)
        else:  # plotting a boolean mask
            slices = slices.to(torch.uint8)
        wandb.log({tag: wandb.Image(make_grid(slices, nrow=4))}, step=iteration)

    # log 12 slices 3x4 of a random sample in a batch
    def log_images(
        self,
        batch: Tensor,
        batch_min: Tensor,
        masks: Tensor,
        iteration: int,
        stage: str,
        tag: str,
        reconstructions: Optional[Tensor] = None,
    ):
        sample_idx = self.rng.integers(len(batch))
        sample, sample_min = batch[sample_idx], batch_min[sample_idx]
        slice_idxs = sorted(self.rng.choice(sample.shape[-1], size=12, replace=False))
        tag = f"{stage}/images/{tag}"
        self._log_images_grid(sample, slice_idxs, tag, iteration, sample_min=sample_min)
        self._log_images_grid(masks[sample_idx], slice_idxs, f"{tag}_mask", iteration)
        if reconstructions is not None:
            self._log_images_grid(
                reconstructions[sample_idx],
                slice_idxs,
                f"{tag}_recon",
                iteration,
                sample_min=sample_min,
            )
