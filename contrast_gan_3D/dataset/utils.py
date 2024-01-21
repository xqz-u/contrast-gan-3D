from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from contrast_gan_3D.utils import io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


def create_ostia_dataframe(
    ostia_files: List[Union[Path, str]],
    ostia_sheet_savename: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    n_ostias = len(ostia_files) * 2  # both ostia in same file
    ostias, datapoint_names = np.zeros((n_ostias, 3), dtype=int), []

    for i, ostia_file in zip(range(0, n_ostias, 2), ostia_files):
        ostias[i : i + 2] = io_utils.load_mevis_coords(ostia_file)[0]
        datapoint_names.append(io_utils.stem(ostia_file.parent))
    logger.info("Total L/R ostia coordinates: %s", ostias.shape)

    ostia_df = []
    for i, name in zip(range(0, len(ostias), 2), datapoint_names):
        for j in [i, i + 1]:
            ostia_df.append({"ID": name} | dict(zip(list("xyz"), ostias[j])))
    ostia_df = pd.DataFrame(ostia_df)

    if ostia_sheet_savename is not None:
        ostia_sheet_savename = str(ostia_sheet_savename)
        if not ostia_sheet_savename.endswith(".xlsx"):
            ostia_sheet_savename += ".xlsx"
        ostia_df.to_excel(ostia_sheet_savename, index=False)
        logger.info("Saved ostia world coordinates to '%s'", ostia_sheet_savename)

    return ostia_df
