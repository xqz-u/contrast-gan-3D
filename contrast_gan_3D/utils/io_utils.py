from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import SimpleITK as sitk


def load_ASOCA_annotated_centerlines(annotation_fname: Union[str, Path]) -> np.ndarray:
    with open(annotation_fname) as fd:
        centerlines = [list(map(float, line.strip().split()[1:])) for line in fd]
    return np.vstack(centerlines)


# works both with .mhd and .nii.gz files
def load_sitk_image(
    image_path: Union[Path, str], swap: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    image = sitk.ReadImage(image_path)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()
    image = sitk.GetArrayFromImage(image)
    if swap:
        image = np.swapaxes(image, 0, 2)  # make channel-first
    return image, {"spacing": np.array(spacing), "offset": np.array(offset)}


def load_h5_image(
    image_path: Union[Path, str]
) -> Tuple[h5py.Dataset, Dict[str, np.ndarray]]:
    content = h5py.File(image_path)
    dataset = content["ccta"]["ccta"]
    return dataset, {
        "spacing": np.array(dataset.attrs["spacing"]),
        "offset": np.array(dataset.attrs["offset"]),
        "centerlines": dataset.attrs["ctl_points"],
    }


def load_centerlines(folder_path: Union[str, Path], create: bool = True) -> np.ndarray:
    folder_path = Path(folder_path)
    vessel_files = folder_path.glob("vessel[0-9]*.txt")
    centerlines = [np.loadtxt(v) for v in vessel_files]
    centerlines = np.concatenate(centerlines or [[]])
    if create and len(centerlines):
        savepath = folder_path / "centerlines.txt"
        np.savez_compressed(savepath, centerlines=centerlines)
        print(f"Saved centerlines to {str(savepath)} with key 'centerlines'")
    return centerlines
