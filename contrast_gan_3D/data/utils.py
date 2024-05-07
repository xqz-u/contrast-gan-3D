from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    load_pickle,
    write_pickle,
)

from contrast_gan_3D.alias import Array
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils, logging_utils

logger = logging_utils.create_logger(name=__name__)


def create_patient(
    ccta_path: str | Path,
    centerlines_dir: str | Path,
    ostia_path: str | Path,
    out_dir: str | Path,
):
    img, meta = io_utils.load_sitk_image(ccta_path)  # img: WHD
    ostia_world, _ = io_utils.load_mevis_coords(ostia_path)  # (2, [xyz])
    centerlines_world = io_utils.load_centerlines(centerlines_dir)  # (N, [xyzr])
    centerlines_mask = geom.world_to_grid_coords(
        centerlines_world[..., :3], meta["offset"], meta["spacing"], img.shape
    )  # WHD
    # stack 3D scan & mask into one 4D array
    scan_and_mask = np.stack([img, centerlines_mask], axis=-1)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_name = io_utils.stem(ccta_path)

    np.save(out_dir / f"{patient_name}.npy", scan_and_mask)
    meta = meta | {
        "ostia_world": ostia_world,
        "centerlines_world": centerlines_world,
        "name": patient_name,
    }
    write_pickle(meta, out_dir / f"{patient_name}_meta.pkl")


def load_patient(patient_name: str) -> tuple[np.ndarray, dict]:
    patient = np.load(patient_name + ".npy", mmap_mode="r+")
    meta = load_pickle(patient_name + "_meta.pkl")
    return patient, meta


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
