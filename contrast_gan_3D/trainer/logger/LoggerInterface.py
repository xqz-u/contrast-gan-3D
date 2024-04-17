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
        scan_types: List[ScanType],
        iteration: int,
        stage: str,
    ):
        ...

    def end_hook(self):
        ...


@dataclass
class SingleThreadedLogger(LoggerInterface):
    def __call__(
        self,
        batches: List[dict],
        reconstructions: List[Optional[Tensor]],
        attenuations: List[Optional[Tensor]],
        scan_types: List[ScanType],
        iteration: int,
        stage: str,
    ):
        for batch, scan_type, recon, attn_map in zip(
            batches, scan_types, reconstructions, attenuations
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
    started_threads: List[Thread] = field(default_factory=list, init=False, repr=False)

    def __call__(
        self,
        batches: List[dict],
        reconstructions: List[Optional[Tensor]],
        attenuations: List[Optional[Tensor]],
        scan_types: List[ScanType],
        iteration: int,
        stage: str,
    ):
        # detach from GPU to avoid threads holding references
        for cont in [reconstructions, attenuations]:
            for i in range(len(cont)):
                cont[i] = to_CPU(cont[i])

        t = Thread(
            target=SingleThreadedLogger.__call__,
            args=(
                self,
                batches,
                reconstructions,
                attenuations,
                scan_types,
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
