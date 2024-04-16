from functools import partial
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
from contrast_gan_3D.model.loss import WassersteinLoss, ZNCCLoss
from contrast_gan_3D.model.utils import wgan_gradient_penalty
from contrast_gan_3D.trainer.logger.LoggerInterface import (
    MultiThreadedLogger,
    SingleThreadedLogger,
)
from contrast_gan_3D.trainer.utils import find_latest_checkpoint
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)

# TODO ask Roel: upsampling as part of generator or only inference hack?
# TODO evaluation pipeline + dump path of scans used for train/test in cval
# TODO reproduce Roel's work so it's comparable in the evaluation
# TODO try better ResNet blocks + initialize from pretrained

# TODO exception handling LoggerInterface

# TODO use fold index to group cval runs belonging to same experiment together
# TODO save folds configuration so that restarting a wandb.run is guaranteed to
#      use the same train data as before it was stopped / save it into wandb itself!

# TODO wrong validation metrics ?
# TODO AMP, DDP ?


class Trainer:
    def __init__(
        self,
        train_iterations: int,
        val_iterations: int,
        validate_every: int,
        train_generator_every: int,
        log_every: int,
        log_images_every: int,
        generator_class: partial,
        critic_class: partial,
        generator_optim_class: partial,
        critic_optim_class: partial,
        hu_loss_instance: nn.Module,
        logger_interface: Union[SingleThreadedLogger, MultiThreadedLogger],
        checkpoint_dir: Union[str, Path],
        device: torch.device,
        weight_clip: Optional[float] = None,
        generator_lr_scheduler_class: Optional[partial] = None,
        critic_lr_scheduler_class: Optional[partial] = None,
        hu_loss_weight: float = 1.0,
        sim_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        gp_weight: float = 10,
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
        self.gp_w = gp_weight
        self.weight_clip = weight_clip

        self.generator: nn.Module = generator_class().to(device)
        self.optimizer_G: Optimizer = generator_optim_class(self.generator.parameters())
        self.lr_scheduler_G = generator_lr_scheduler_class
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G: LRScheduler = self.lr_scheduler_G(self.optimizer_G)

        self.critic: nn.Module = critic_class().to(device)
        self.optimizer_D: Optimizer = critic_optim_class(self.critic.parameters())
        self.lr_scheduler_D = critic_lr_scheduler_class
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D: LRScheduler = self.lr_scheduler_D(self.optimizer_D)

        self.loss_GAN = WassersteinLoss()
        self.loss_similarity = ZNCCLoss()
        self.loss_HU = hu_loss_instance
        self.logger_interface = logger_interface

        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.iteration = 0
        self.load_checkpoint(find_latest_checkpoint(self.checkpoint_dir))

    def train_critic(
        self, real: Tensor, reconstructions: Tensor, retain_graph: bool
    ) -> Dict[str, Tensor]:
        self.optimizer_D.zero_grad(set_to_none=True)

        real_logits: Tensor = self.critic(real)
        # detach() avoids computing gradients wrt generator's output
        fake_logits: Tensor = self.critic(reconstructions.detach())

        # critic goal: max E[critic(real)] - E[critic(fake)] <-> min E[critic(fake)] - E[critic(real)]
        loss_critic: Tensor = self.gan_loss_w * self.loss_GAN(fake_logits, real_logits)
        if self.weight_clip is None:
            loss_critic += wgan_gradient_penalty(
                real.repeat((2,) + (1,) * len(real.shape[1:])),
                reconstructions,
                self.critic,
                device=self.device,
                lambda_=self.gp_w,
            )

        loss_critic.backward(retain_graph=retain_graph and self.weight_clip is None)
        self.optimizer_D.step()
        if self.weight_clip is not None:
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.step()

        real_logits = real_logits.repeat((2,) + (1,) * len(real_logits.shape[1:]))
        return {"D": loss_critic, "D-real-fake-delta": real_logits - fake_logits}

    def train_generator(
        self, inputs: Tensor, reconstructions: Tensor, centerlines_masks: Tensor
    ) -> Dict[str, Tensor]:
        self.optimizer_G.zero_grad(set_to_none=True)

        # generator goal: max E[critic(fake)] <-> min -E[critic(fake)]
        loss_G = self.gan_loss_w * -self.loss_GAN(self.critic(reconstructions))
        loss_sim = self.sim_loss_w * self.loss_similarity(reconstructions, inputs)
        loss_hu = self.hu_loss_w * self.loss_HU(reconstructions, centerlines_masks)
        full_loss_G: Tensor = loss_G + loss_sim + loss_hu

        full_loss_G.backward()
        self.optimizer_G.step()
        if self.lr_scheduler_G is not None:
            self.lr_scheduler_G.step()

        return {"G": loss_G, "G-full": full_loss_G, "sim": loss_sim, "HU": loss_hu}

    def train_step(self, patches: List[dict], iteration: int):
        opt, low, high = patches
        opt: Tensor = opt["data"].to(self.device, non_blocking=True)
        subopt = torch.cat([low["data"], high["data"]])
        subopt = subopt.to(self.device, non_blocking=True)

        # generate optimal image
        attenuation: Tensor = self.generator(subopt)
        opt_hat: Tensor = subopt - attenuation

        do_train_generator = iteration % self.train_generator_every == 0
        # when gradient penalty is used and `opt_hat` is the same in consecutive
        # critic/generator updates, critic and generator loss share parts of the
        # same computational graph, meaning `retain_graph` must be True on the
        # first of the two loss.backward()
        log_dict = self.train_critic(opt, opt_hat, do_train_generator)
        if do_train_generator:
            subopt_mask = torch.cat([low["seg"], high["seg"]])
            subopt_mask = subopt_mask.to(self.device, non_blocking=True)
            log_dict |= self.train_generator(subopt, opt_hat, subopt_mask)

        # ------------------ logging
        if iteration % self.log_every == 0:
            self.logger_interface.logger.log_loss(
                {k: v.mean() for k, v in log_dict.items()}, iteration, "train"
            )

        if iteration % self.log_images_every == 0:
            self.logger_interface(
                patches,
                [None, *opt_hat.chunk(2)],
                [None, *attenuation.chunk(2)],
                list(ScanType),
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
        self.critic.train()
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
                and iteration != 0
                and iteration % self.checkpoint_every == 0
            ):
                self.save_checkpoint(iteration)

            if profiler:
                profiler.step()

        if profiler:
            profiler.stop()
        if self.checkpoint_every is not None:
            self.save_checkpoint(self.train_iterations)
        self._manage_augmenters([train_loaders, val_loaders], "end")
        self.logger_interface.end_hook()

    def validate(self, val_loaders: Dict[int, BGenAugmenter], train_iteration: int):
        self.critic.eval()
        self.generator.eval()
        self.retrieve_batch_size(val_loaders[ScanType.OPT.value], "val")

        (
            loss_sim,
            loss_G,
            loss_real_D,
            loss_fake_D,
            D_real_tot,
            D_fake_tot,
        ) = torch.zeros(6, dtype=torch.float32, device=self.device)
        loggable = []

        with torch.no_grad():
            for i, scan_type_enum in tqdm_product(
                range(self.val_iterations),
                ScanType,
                desc=f"Val {train_iteration // self.val_every}",
            ):
                batch = next(val_loaders[scan_type_enum.value])
                sample = batch["data"].to(self.device, non_blocking=True)

                if scan_type_enum == ScanType.OPT:
                    D_real = self.critic(sample)

                    loss_real_D -= self.loss_GAN(D_real)
                    D_real_tot += D_real.sum()
                else:
                    attenuation = self.generator(sample)
                    sample_hat = sample - attenuation
                    D_fake = self.critic(sample_hat)
                    batch_loss_G = self.loss_GAN(D_fake)

                    D_fake_tot += D_fake.sum()
                    loss_fake_D += batch_loss_G
                    loss_G -= batch_loss_G
                    loss_sim += self.loss_similarity(sample_hat, sample)

                if i == 0 and scan_type_enum != ScanType.OPT:
                    loggable.append([batch, sample_hat, attenuation])
                    if len(loggable) == (len(ScanType) - 1):
                        patches, reconstructions, attenuations = list(zip(*loggable))
                        self.logger_interface(
                            patches,
                            list(reconstructions),
                            list(attenuations),
                            list(ScanType)[1:],
                            train_iteration,
                            "validation",
                        )
                        # eliminate references to free up GPU memory
                        for s, r, a in loggable:
                            del s, r, a
                        loggable = None

        self.critic.train()
        self.generator.train()

        val_loss = {
            "D": (loss_real_D + loss_fake_D) / self.val_tot_samples,
            "G": loss_G / self.val_subopt_samples,
            "sim": loss_sim / self.val_subopt_samples,
            "D-real-fake-delta": (D_real_tot - D_fake_tot) / self.val_tot_samples,
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

    def save_checkpoint(self, iteration: int):
        state = {"iteration": iteration}
        for attr in self.model_torch_attrs:
            el = getattr(self, attr, None)
            state[attr] = el if el is None else el.state_dict()
        torch.save(state, self.checkpoint_dir / f"{iteration}.pt")
        logger.info("Checkpoint iteration %d", iteration)

    def load_checkpoint(self, ckpt_path: Optional[Path]):
        if ckpt_path is not None and ckpt_path.is_file():
            logger.info("Resuming run from '%s'", str(ckpt_path))
            checkpoint: dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
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
            if stage == "val":
                # assume dataloaders with same batch size across classes
                self.val_opt_samples = self.val_iterations * self.val_bs
                self.val_subopt_samples = 2 * self.val_opt_samples
                self.val_tot_samples = self.val_opt_samples + self.val_subopt_samples
