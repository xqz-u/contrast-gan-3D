from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import trange

from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.model.loss import HULoss, WassersteinLoss, ZNCCLoss
from contrast_gan_3D.trainer.Reloader import Reloader


# TODO check with batchviewer that patches are correct
# TODO proper augmentations
# TODO restart mechanism
# TODO proper logging
# TODO collection of injected attributes in a dict for experiment tracking
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
        generator_lr_scheduler: Optional[LRScheduler] = None,
        discriminator_lr_scheduler: Optional[LRScheduler] = None,
        hu_loss_weight: float = 1.0,
        sim_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        device_num: Optional[int] = None,
        **hu_loss_kwargs,
    ):
        device_str = "cuda"
        if device_num is not None:
            device_str += f":{device_num}"
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        self.hu_loss_w = hu_loss_weight
        self.sim_loss_w = sim_loss_weight
        self.gan_loss_w = gan_loss_weight

        self.train_generator_every = train_generator_every

        self.generator = generator
        self.optimizer_G = generator_optim
        if generator_lr_scheduler is not None:
            self.scheduler_G = generator_lr_scheduler

        self.discriminator = discriminator
        self.optimizer_D = discriminator_optim
        if discriminator_lr_scheduler is not None:
            self.scheduler_D = discriminator_lr_scheduler

        self.loss_GAN = WassersteinLoss()
        self.loss_similarity = ZNCCLoss()
        self.loss_HU = HULoss(**hu_loss_kwargs)

        self.start_iteration = 0
        self.logger = None

    def train_step(
        self,
        opt: torch.Tensor,
        subopt: torch.Tensor,
        subopt_mask: torch.Tensor,
        iteration: int,
    ) -> Dict[str, float]:
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
            if (scheduler := getattr(self, f"scheduler_{tag}")) is not None:
                scheduler.step()

        return ret

    def fit(
        self,
        train_iterations: int,
        train_loaders: Dict[int, Reloader],
        val_loaders: Dict[int, Reloader],
    ):
        self.generator.to(self.device).train()
        self.discriminator.to(self.device).train()

        # writer = SummaryWriter(log_dir=self.logs_dir)

        for iteration in trange(self.start_iteration, train_iterations):
            opt = next(train_loaders[ScanType.OPT.value])
            low = next(train_loaders[ScanType.LOW.value])
            high = next(train_loaders[ScanType.HIGH.value])
            # to GPU
            for el in [opt, low, high]:
                for k in ["data", "seg"]:
                    el[k] = el[k].to(self.device)

            subopt = torch.cat([low["data"], high["data"]])
            subopt_mask = torch.cat([low["seg"], high["seg"]])

            train_loss = self.train_step(opt["data"], subopt, subopt_mask, iteration)

            # if iteration % 100 == 0:
            if iteration % 2 == 0:
                self.log_loss(train_loss, iteration, "train")
                # self.log_loss(writer, [loss_i, loss_a, loss_D], iteration, "train")

            # if iteration % 200 == 0:
            if iteration % 2:
                # self.log_images(writer, subopt, opt_hat, iteration, "images_train/fake")
                # self.log_images(writer, opt, opt, iteration, "images_train/real")
                self.log_loss(self.validate(val_loaders), iteration, "validation")
                # self.log_loss(writer, [loss_i, loss_a, loss_D], iteration, "val")

            if iteration % 1000 == 0:
                ...
                # self.save_train_state(iteration)

    def validate(self, val_loaders: Dict[int, Reloader]) -> Dict[str, float]:
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
                            ...
                            # self.log_images(
                            #     writer, batch, batch, iteration, scan_type.name
                            # )
                    else:
                        sample_hat = sample - self.generator(sample)
                        adversarial_loss = self.loss_GAN(self.discriminator(sample_hat))
                        loss_fake += adversarial_loss
                        loss_a -= adversarial_loss
                        loss_i += self.loss_similarity(sample_hat, sample)
                        if i == 0:
                            ...
                            # self.log_images(
                            #     writer, batch, batch_hat, iteration, scan_type.name
                            # )

        n_opt = len(val_loaders[ScanType.OPT.value].dataset)
        n_low = len(val_loaders[ScanType.LOW.value].dataset)
        n_high = len(val_loaders[ScanType.HIGH.value].dataset)
        n_subopt = n_low + n_high

        ret = {
            "n_opt": n_opt,
            "n_subopt": n_subopt,
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

    @staticmethod
    def log_loss(loss_dict: Dict[str, float], iteration: int, stage: str):
        print(f"\033[1m{stage}\033[0m iteration", iteration, "loss:")
        for k, v in loss_dict.items():
            print(f"\t{k}: {v}")
        print()
