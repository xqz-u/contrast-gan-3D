from dataclasses import dataclass, field
from threading import Thread
from typing import List, Optional

from torch import Tensor

from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.trainer.logger.WandbLogger import WandbLogger
from contrast_gan_3D.utils import logging_utils, now_str, to_CPU

logger = logging_utils.create_logger(name=__name__)


@dataclass
class LoggerInterface:
    logger: WandbLogger

    def __call__(
        self,
        batches: List[dict],
        reconstructions: List[Optional[Tensor]],
        attenuations: List[Optional[Tensor]],
        iteration: int,
        stage: str,
    ):
        ...

    def end_hook(self):
        ...


# TODO thread exception handling
@dataclass
class SingleThreadedLogger(LoggerInterface):
    def __call__(
        self,
        batches: List[dict],
        reconstructions: List[Optional[Tensor]],
        attenuations: List[Optional[Tensor]],
        iteration: int,
        stage: str,
    ):
        for batch, scan_type, recon, attn_map in zip(
            batches, ScanType, reconstructions, attenuations
        ):
            buffer = self.logger(
                batch["data"],
                iteration,
                stage,
                scan_type.name,
                batch["name"],
                masks=batch["seg"],
                reconstructions=to_CPU(recon),
                attenuations=to_CPU(attn_map),
            )
            self.logger.log_images(buffer)


@dataclass
class MultiThreadedLogger(LoggerInterface):
    started_threads: List[Thread] = field(default_factory=list, init=False)

    def __call__(
        self,
        batches: List[dict],
        reconstructions: List[Optional[Tensor]],
        attenuations: List[Optional[Tensor]],
        iteration: int,
        stage: str,
    ):
        # make sure to free GPU before passing reference onto thread
        reconstructions[1] = to_CPU(reconstructions[1])
        reconstructions[2] = to_CPU(reconstructions[2])
        attenuations[1] = to_CPU(attenuations[1])
        attenuations[2] = to_CPU(attenuations[2])

        t = Thread(
            target=SingleThreadedLogger.__call__,
            args=(
                self,
                batches,
                reconstructions,
                attenuations,
                iteration,
                stage,
            ),
            name=f"logger-{stage}-{iteration}",
        )
        self._start(t)

    def end_hook(self):
        for t in self.started_threads:
            if t.is_alive():
                logger.info("Waiting on thread '%s'", t.name)
                t.join()

    def _start(self, t: Thread):
        logger.debug("%s: '%s'", now_str(), t.name)
        t.start()
        self.started_threads.append(t)
