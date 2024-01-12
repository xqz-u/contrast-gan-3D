import logging
import sys
from typing import Optional, Union


def create_logger(
    log_level: Union[str, int] = "INFO", name: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name or __name__)

    logger.setLevel(log_level)  # NOTE if int, `log_level` should be well-defined

    handler = logging.StreamHandler(stream=sys.stdout)
    fmt_str = "[%(asctime)s: %(levelname)s] %(message)s (%(name)s:%(lineno)s)"
    handler.setFormatter(logging.Formatter(fmt=fmt_str))
    logger.addHandler(handler)

    return logger


# NOTE call last after all modules defining loggers have been imported
def set_project_loggers_level(level: Union[str, int] = "INFO"):
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith("contrast_gan_3D") and isinstance(logger, logging.Logger):
            old_level = logging.getLevelName(logger.level)
            logger.setLevel(level)
            print(f"{name}: {old_level} -> {level}")
