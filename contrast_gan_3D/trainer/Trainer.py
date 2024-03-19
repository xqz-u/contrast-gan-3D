from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from matplotlib import cm, colors, figure
from matplotlib import pyplot as plt
from torch import Tensor, nn, profiler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import make_grid
from tqdm.auto import trange

import wandb
from contrast_gan_3D.alias import BGenAugmenter, ScanType
from contrast_gan_3D.config import CHECKPOINTS_DIR
from contrast_gan_3D.constants import VMAX, VMIN
from contrast_gan_3D.data.utils import MinMaxNormShift, minmax_denorm, minmax_norm
from contrast_gan_3D.model.loss import WassersteinLoss, ZNCCLoss
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)

# TODO plot discriminator/generator losses on real/fake samples together & separately

# TODO use fold index to group cval runs belonging to same experiment together
# TODO save folds configuration so that restarting a wandb.run is guaranteed to
#      use the same train data as before it was stopped / save it into wandb itself!

# TODO profile
# TODO log images in background thread

# TODO `margin` parameter of batchgenerator's crop?
# TODO create few 3D scans of fake images to import into ITK-SNAP

# TODO check wait time of augmenters is tuned right

# TODO AMP, DDP (?)

# TODO other generator/discriminator architectures
# TODO gradient penalized Wasserstein loss instead of clipping network
#      parameters

# TODO inference: patches aggregation

# TODO run centerline extraction + opt/low/high dataset creation with new HU values
#      shifting approach


class Trainer:
    def __init__(
        self,
        train_iterations: int,
        val_iterations: int,
        validate_every: int,
        train_generator_every: int,
        log_every: int,
        log_images_every: int,
        val_bs: int,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optim: Optimizer,
        discriminator_optim: Optimizer,
        max_HU_delta: int,
        hu_loss_instance: nn.Module,
        normalizer: MinMaxNormShift,
        run_id: str,
        device: torch.device,
        generator_lr_scheduler: Optional[LRScheduler] = None,
        discriminator_lr_scheduler: Optional[LRScheduler] = None,
        hu_loss_weight: float = 1.0,
        sim_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        checkpoint_every: int = 1000,
        rng: Optional[np.random.Generator] = None,
        profiler_dir: Optional[Union[Path, str]] = None,
    ):
        self.device = device
        logger.info("Using device: %s", self.device)
        self.rng = rng or np.random.default_rng()

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
        self.max_HU_delta = max_HU_delta
        self.normalizer = normalizer

        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = CHECKPOINTS_DIR / f"{run_id}.pt"
        self.checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        self.iteration = 0
        self.load_checkpoint(self.checkpoint_path)

        self.val_bs = val_bs

        self.profiler = None
        if profiler_dir is not None:
            # hard-set to show cost of wandb logging
            self.val_every, self.val_iterations = 20, 10
            self.log_every = self.log_images_every = 20
            logger.info(
                "PyTorch profiler with TensorBoard trace in: '%s'", profiler_dir
            )
            activities = [profiler.ProfilerActivity.CPU]
            if "cuda" in self.device.type:
                activities.append(profiler.ProfilerActivity.CUDA)
            self.profiler = profiler.profile(
                activities=activities,
                schedule=profiler.schedule(skip_first=5, wait=2, warmup=3, active=10),
                on_trace_ready=profiler.tensorboard_trace_handler(profiler_dir),
                profile_memory=True,
                with_stack=True,
            )

    def train_step(self, patches: List[dict], iteration: int):
        self.optimizer_D.zero_grad(set_to_none=True)

        opt, low, high = patches
        subopt = torch.cat([low["data"], high["data"]])
        subopt_mask = torch.cat([low["seg"], high["seg"]])
        # to GPU
        opt, subopt, subopt_mask = [
            el.to(self.device, non_blocking=True)
            for el in [opt["data"], subopt, subopt_mask]
        ]

        # generate optimal image
        attenuation_map: Tensor = self.generator(subopt)
        opt_hat: Tensor = subopt - self.normalizer(attenuation_map * self.max_HU_delta)

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

            loss_G = self.gan_loss_w * -self.loss_GAN(self.discriminator(opt_hat))
            loss_sim = self.sim_loss_w * self.loss_similarity(opt_hat, subopt)
            loss_hu = self.hu_loss_w * self.loss_HU(opt_hat, subopt_mask)
            if torch.isnan(loss_hu):
                loss_hu = 0.0

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
            self.log_loss(
                {k: v.mean() for k, v in log_dict.items()}, iteration, "train"
            )

        # return opt_hat, attenuation_map

        if iteration % self.log_images_every == 0:
            # reconstruction and low/high are logged with the same indices
            for data, scan_type, recon, attn_map in zip(
                patches,
                ScanType,
                [None, *opt_hat.chunk(2)],
                [None, *attenuation_map.chunk(2)],
            ):
                self.log_images(
                    data["data"],
                    iteration,
                    "train",
                    scan_type.name,
                    data["name"],
                    masks=data["seg"],
                    reconstructions=recon,
                    attenuations=attn_map,
                )

    def fit(
        self,
        train_loaders: Dict[int, BGenAugmenter],
        val_loaders: Dict[int, BGenAugmenter],
    ):
        self.generator.train()
        self.discriminator.train()
        # start batchgenerator's augmentations asynchronously
        self._manage_augmenters([train_loaders, val_loaders], "start")

        for iteration in trange(
            self.iteration, self.train_iterations, desc="Train", unit="batch"
        ):
            if self.profiler:
                if iteration + 1 >= 65:
                    break
                self.profiler.step()

            # NOTE order is determined by ScanType
            patches = [next(train_loaders[scan_type.value]) for scan_type in ScanType]
            self.train_step(patches, iteration)

            if iteration != 0 and iteration % self.val_every == 0:
                self.validate(val_loaders, iteration)

            if iteration % self.checkpoint_every == 0:
                self.save_checkpoint(self.checkpoint_path, iteration)

        if self.profiler:
            self.profiler.stop()
        self.save_checkpoint(self.checkpoint_path, self.train_iterations)
        self._manage_augmenters([train_loaders, val_loaders], "end")

    def validate(self, val_loaders: Dict[int, BGenAugmenter], train_iteration: int):
        self.discriminator.eval()
        self.generator.eval()

        loss_sim, loss_G, loss_real_D, loss_fake_D = torch.zeros(
            4, dtype=torch.float32, device=self.device
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
                    sample_hat, attenuation_map = None, None

                    if scan_type == ScanType.OPT:
                        loss_real_D -= self.loss_GAN(self.discriminator(sample))
                    else:
                        attenuation_map = self.generator(sample)
                        sample_hat = sample - self.normalizer(
                            attenuation_map * self.max_HU_delta
                        )
                        batch_loss_G = self.loss_GAN(self.discriminator(sample_hat))
                        loss_fake_D += batch_loss_G
                        loss_G -= batch_loss_G
                        loss_sim += self.loss_similarity(sample_hat, sample)

                    if i == 0:
                        self.log_images(
                            sample,
                            train_iteration,
                            "validation",
                            scan_type.name,
                            batch["name"],
                            reconstructions=sample_hat,
                            attenuations=attenuation_map,
                        )

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
        self.log_loss(val_loss, train_iteration, "validation")

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
    def log_loss(loss_dict: Dict[str, Tensor], iteration: int, stage: str):
        wandb.log({f"{stage}/{k}": v for k, v in loss_dict.items()}, step=iteration)

    def _log_images_grid(
        self,
        slices: Union[Tensor, np.ndarray],
        tag: str,
        iteration: int,
        fig: figure.Figure = None,
        caption: Optional[str] = None,
        **grid_args,
    ):
        if isinstance(slices, np.ndarray):
            slices = torch.from_numpy(slices)
        # CHWD -> DCHW -> C,HxD,WxD
        grid = make_grid(slices.permute(3, 0, 1, 2), **grid_args)
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            ax, *_ = fig.get_axes()
        ax.imshow(grid.permute(1, 2, 0))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.tight_layout()
        wandb.log({tag: wandb.Image(ax, caption=caption)}, step=iteration)
        plt.close(fig)

    # log 64 slices of a random sample in a batch
    def log_images(
        self,
        scans: Tensor,
        iteration: int,
        stage: str,
        scan_type: str,
        names: List[str],
        masks: Optional[Tensor] = None,
        reconstructions: Optional[Tensor] = None,
        attenuations: Optional[Tensor] = None,
    ):
        value_range = (self.normalizer.low, self.normalizer.high)
        sample_idx = self.rng.integers(len(scans))
        slice_idxs = sorted(self.rng.choice(scans.shape[-1], size=64, replace=False))
        indexer = [sample_idx, ..., slice_idxs]
        grid_args = {"normalize": True, "value_range": (VMIN, VMAX)}
        cmap, fig, caption = cm.RdBu, None, names[sample_idx]
        caption_cp = caption

        if stage == "train":
            # show centerlines by scattering manually
            ctls = masks[indexer].permute(3, 0, 1, 2).to(torch.float16)
            ctls_grid = make_grid(ctls)
            # DHW -> HWD (yxz)
            cart = geom.grid_to_cartesian_coords(ctls_grid.cpu().permute(1, 2, 0))
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                cart[:, 1],
                cart[:, 0],
                c="red",
                s=plt.rcParams["lines.markersize"] * 0.8,
            )
            caption_cp = f"{caption} {np.prod(cart.shape)}/{np.prod(masks[sample_idx].shape)} centerlines"

        slices = minmax_denorm(scans[indexer] + self.normalizer.shift, value_range)
        workspace = f"{stage}/images/{scan_type}"
        self._log_images_grid(
            slices.cpu(),
            f"{workspace}/sample",
            iteration,
            fig=fig,
            caption=caption_cp,
            **grid_args,
        )
        if reconstructions is not None:
            recon = reconstructions[indexer]
            recon = minmax_denorm(recon, value_range) - self.normalizer.low
            self._log_images_grid(
                recon.cpu(),
                f"{workspace}/reconstruction",
                iteration,
                caption=caption,
                **grid_args,
            )
        if attenuations is not None:
            # normalize [-1, 1]->[0, 1] for colormap (min and max from entire sample)
            attn_sample = attenuations[sample_idx]
            low, high = attn_sample.min().item(), attn_sample.max().item()
            attn = minmax_norm(attenuations[indexer], (low, high))
            # https://discuss.pytorch.org/t/torch-utils-make-grid-with-cmaps/107471/2
            attn = np.apply_along_axis(cmap, 0, attn.detach().cpu().numpy()).squeeze()
            # add colorbar
            fig, ax = plt.subplots(figsize=(10, 10))
            mappable = cm.ScalarMappable(norm=colors.Normalize(low, high), cmap=cmap)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.8)
            cbar.set_ticks(np.linspace(low, high, 5))
            self._log_images_grid(
                attn, f"{workspace}/attenuation", iteration, fig=fig, caption=caption
            )

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
