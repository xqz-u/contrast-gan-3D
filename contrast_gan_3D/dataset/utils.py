from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from contrast_gan_3D.utils import geometry, io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


# NOTE from Nils' code
def load_mevis_coords(sourcefile: Union[Path, str]) -> Tuple[np.ndarray, np.ndarray]:
    def add_to_array(line: str, tag: str, arr: np.ndarray, idx: int) -> int:
        if tag in line:
            els = line.replace(f"<{tag}>", "").replace(f"</{tag}>", "").split()
            arr[idx] = list(map(float, els))[:3]
            idx += 1
        return idx

    points = np.zeros((1, 1), dtype=np.float32)
    vecs = np.zeros((1, 1), dtype=np.float32)
    pindex, vindex = 0, 0
    with open(sourcefile, "r") as f:
        for line in f:
            line = line.strip()
            if "ListSize" in line:
                nitems = int(line.replace("<ListSize>", "").replace("</ListSize>", ""))
                points.resize((nitems, 3))
                vecs.resize((nitems, 3))
            pindex = add_to_array(line, "pos", points, pindex)
            vindex = add_to_array(line, "vec", vecs, vindex)
    return points, vecs


def create_ostia_dataframe(
    ostia_files: List[Union[Path, str]],
    ccta_files: List[Union[Path, str]],
    ostia_sheet_savename: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    n_ostias = len(ostia_files) * 2
    ostias, datapoint_names = np.zeros((n_ostias, 3), dtype=int), []

    for i, h5_image_path, ostia_file in zip(
        range(0, n_ostias, 2), ccta_files, ostia_files
    ):
        ostia_world_coords, _ = load_mevis_coords(ostia_file)

        with h5py.File(h5_image_path) as fd:
            ccta = fd["ccta"]["ccta"]
            offset, spacing = ccta.attrs["offset"], ccta.attrs["spacing"]
            ostia_img_coords = geometry.world_to_image_coords(
                ostia_world_coords, offset, spacing
            )

        ostias[i : i + 2] = ostia_img_coords
        datapoint_names.append(io_utils.stem(ostia_file.parent))
    logger.info("Total L/R ostia coordinates: %s", ostias.shape)

    ostia_df = []
    for i, name in zip(range(0, len(ostias), 2), datapoint_names):
        for j in [i, i + 1]:
            ostia_df.append({"name": name} | dict(zip(list("xyz"), ostias[j])))
    ostia_df = pd.DataFrame(ostia_df)

    if ostia_sheet_savename is not None:
        ostia_sheet_savename = str(ostia_sheet_savename)
        if not ostia_sheet_savename.endswith(".xlsx"):
            ostia_sheet_savename += ".xlsx"
        ostia_df.to_excel(ostia_sheet_savename)
        logger.info("Saved ostia coordinates to '%s'", ostia_sheet_savename)

    return ostia_df
