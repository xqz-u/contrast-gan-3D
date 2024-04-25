from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from patchly.aggregator import Aggregator
from patchly.sampler import GridSampler
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from contrast_gan_3D.alias import Array, ArrayShape
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.data.Scaler import Scaler
from contrast_gan_3D.eval.CCTAEvalDataset import CCTAEvalDataset2D, CCTAEvalDataset3D
from contrast_gan_3D.model.utils import compute_convolution_filters_shape
from contrast_gan_3D.utils import io_utils
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


@dataclass
class CCTAContrastCorrector:
    model: Callable[[], nn.Module]
    scaler: Scaler
    device: torch.device
    inference_patch_size: Optional[ArrayShape] = None
    checkpoint_path: Optional[Path] = None
    upsampler: Callable[[Tensor], Tensor] = field(init=False, default=lambda _, x: x)

    def __post_init__(self):
        self.model: nn.Module = self.model()
        if self.checkpoint_path is not None:
            self.load_model(self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.correct_scan = self.correct_scan_3D
        if self.inference_patch_size is None or len(self.inference_patch_size) < 3:
            self.correct_scan = self.correct_scan_2D
            self.inference_patch_size = (512, 512)
        model_output_shape = compute_convolution_filters_shape(
            self.model, (1,) + self.inference_patch_size, show=False
        )
        if model_output_shape[1:] != list(self.inference_patch_size):
            logger.info(
                "Inference patch shape %s != model output shape %s, upsample to %s",
                self.inference_patch_size,
                model_output_shape[1:],
                self.inference_patch_size,
            )
            self.upsampler = nn.Upsample(size=self.inference_patch_size)

    def load_model(self, checkpoint_path: Union[str, Path]):
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt["generator"])
        self.checkpoint_path = Path(checkpoint_path)
        logger.info("Loaded model checkpoint '%s'", str(self.checkpoint_path))

    def correct_scan_3D(self, scan: HD5Scan, batch_size: int) -> Tensor:
        sampler = GridSampler(scan.ccta, scan.ccta.shape, self.inference_patch_size)
        loader = DataLoader(
            CCTAEvalDataset3D(self.scaler, sampler),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        aggregator = Aggregator(
            sampler,
            output=torch.empty((1, *scan.ccta.shape), device=self.device),
            spatial_first=False,
            has_batch_dim=True,
            device=self.device,
        )
        for patch, bbox in tqdm(loader, desc=str(scan.path)):
            patch_scaled = patch.to(torch.float32).to(self.device, non_blocking=True)
            corrected = patch_scaled - self.upsampler(self.model(patch_scaled))
            aggregator.append(corrected, bbox)
        return aggregator.get_output()

    def correct_scan_2D(self, scan: HD5Scan, batch_size: int) -> Tensor:
        loader = DataLoader(
            CCTAEvalDataset2D(self.scaler, scan.ccta[::]),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        i, out_shape = 0, (scan.ccta.shape[-1], 1, *scan.ccta.shape[:-1])
        corrected_scan = torch.empty(out_shape, device=self.device)
        for batch_scaled in tqdm(loader, desc=str(scan.path)):
            batch_scaled = batch_scaled.to(self.device, non_blocking=True)
            corrected = batch_scaled - self.upsampler(self.model(batch_scaled))
            corrected_scan[i : i + len(batch_scaled)] = corrected
            i += len(batch_scaled)
        return corrected_scan.permute((1, 2, 3, 0))

    @torch.no_grad
    def __call__(
        self, ccta_path: Union[str, Path], batch_size: int = 16
    ) -> Tuple[Tensor, np.ndarray, np.ndarray]:
        with HD5Scan(ccta_path) as scan:
            corrected_scan = self.correct_scan(scan, batch_size)
        corrected_scan = self.scaler.unscale(corrected_scan).squeeze().detach().cpu()
        return corrected_scan, scan.meta["offset"], scan.meta["spacing"]

    @staticmethod
    def save_scan(
        ccta: Array, offset: np.ndarray, spacing: np.ndarray, savepath: Union[str, Path]
    ):
        if isinstance(ccta, Tensor):
            ccta = ccta.numpy()
        # HWD -> DHW (xyz->zyx, numpy to sitk convention)
        savepath = str(savepath)
        logger.info("Saving scan to '%s'...", savepath)
        io_utils.to_itksnap_volume(ccta.transpose(2, 0, 1), offset, spacing, savepath)
        logger.info("DONE saved '%s'", savepath)
