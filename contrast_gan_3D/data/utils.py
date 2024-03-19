from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch import nn
from tqdm.auto import tqdm

from contrast_gan_3D.alias import Array
from contrast_gan_3D.constants import MAX_HU, MIN_HU
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.utils import io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


def create_ostia_dataframe(
    ostia_files: List[Union[Path, str]],
    ostia_sheet_savename: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    n_ostias = len(ostia_files) * 2  # both ostia in same file
    ostias, datapoint_names = np.zeros((n_ostias, 3), dtype=np.float32), []

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


def label_ccta_scan(
    ostia_HU_df: pd.DataFrame, is_cadrads: bool = True, std_threshold: float = 500.0
) -> pd.DataFrame:
    ret = (
        ostia_HU_df.iloc[
            ostia_HU_df.groupby("ID" if is_cadrads else "id").apply(
                lambda x: x["std"].idxmin()
            )
        ]
        .copy()
        .reset_index(drop=True)
    )
    if is_cadrads:
        ret = ret.drop_duplicates(subset=["mu", "std"])
    ret = ret[ret["std"] < std_threshold]
    # label the CT scans based on the mean HU intensity at the coronary aortic root
    ret.loc[ret["mu"].between(300, 500), "label"] = 0
    ret.loc[ret["mu"] <= 300, "label"] = -1
    ret.loc[ret["mu"] >= 500, "label"] = 1
    ret["label"] = ret["label"].astype("int8")
    return ret


# NOTE if `value_range` is given, values in `x` outside of it should be clipped
# either before or after normalization
def minmax_norm(
    x: Union[Array, float], value_range: Optional[Tuple[float, float]] = None
):
    if value_range is None:
        assert isinstance(x, (np.ndarray, torch.Tensor))
        value_range = (x.min(), x.max())
    low, high = value_range
    return (x - low) / max(high - low, 1e-5)


# pack low, high, shift in one place that still uses autograd
class MinMaxNormShift(nn.Module):
    def __init__(self, low: float, high: float, shift: float):
        super().__init__()
        self.low, self.high, self.shift = low, high, shift

    def forward(self, x: torch.Tensor):
        return minmax_norm(x, (self.low, self.high)) - self.shift


# NOTE use only the train dataset mean, exclude test data! e.g. in cval loop
def compute_dataset_mean(*ct_scan_paths: Iterable[Union[str, Path]]):
    sum_, n_pix = 0, 0
    for image_path in tqdm(ct_scan_paths):
        image_path = Path(image_path)
        if image_path.suffix == ".h5":
            with HD5Scan(image_path) as scan:
                image = scan.ccta[::]
        else:
            # same preprocessing used to create .h5 files, besides orientation adjustment
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype(np.int16)
            diff = image.min() - MIN_HU
            if diff >= abs(MIN_HU):
                image -= diff
            image = np.clip(image, MIN_HU, MAX_HU)
            # sum all voxel values
        sum_ += image.sum()
        n_pix += np.prod(image.shape)
    return sum_ / n_pix
