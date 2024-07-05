from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torchio as tio

from contrast_gan_3D.constants import MAX_HU, MIN_HU, ORIENTATION
from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


def get_scan_orientation(img: Union[sitk.Image, Path]) -> str:
    img = (
        tio.ScalarImage.from_sitk(img)
        if isinstance(img, sitk.Image)
        else tio.ScalarImage(img)
    )
    return "".join(img.orientation)


def basename(path: Union[str, Path]) -> str:
    return str(path).split("/")[-1]


def stem(path: Union[str, Path]) -> str:
    return basename(path).split(".")[0]


def load_centerlines(folder_path: Union[str, Path], glob_str: str = None) -> np.ndarray:
    folder_path = Path(folder_path)
    glob_str = glob_str or "vessel[0-9]*.txt"
    vessel_files = folder_path.glob(glob_str)
    centerlines = [np.loadtxt(v) for v in vessel_files]
    return np.concatenate(centerlines or [[]], axis=0, dtype=np.float32)


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


def load_sitk_image(
    image_path: Union[Path, str],
    segmentation: bool = False,
    target_orientation: str = ORIENTATION,
) -> Tuple[np.ndarray, Dict[str, Union[str, np.ndarray]]]:
    image_path = Path(image_path)
    image = sitk.ReadImage(image_path.resolve())

    orientation = get_scan_orientation(image)
    if orientation != target_orientation:
        image = sitk.DICOMOrient(image, target_orientation)
        new_orientation = get_scan_orientation(image)
        logger.debug(
            "Changed orientation '%s': %s -> %s",
            str(image_path),
            orientation,
            new_orientation,
        )
        orientation = new_orientation

    spacing, offset = image.GetSpacing(), image.GetOrigin()
    image = sitk.GetArrayFromImage(image).transpose(2, 1, 0)  # DHW -> WHD
    logger.debug(
        "Original image dtype %s range (%s, %s)", image.dtype, image.min(), image.max()
    )

    image = image.astype(np.int16)
    if not segmentation:
        logger.debug("Scaling image...")
        # constrain the scan to lie in [MIN_HU, MAX_HU]
        if (diff := image.min() - MIN_HU) >= abs(MIN_HU):
            image -= diff
        image = image.clip(MIN_HU, MAX_HU)

    min_, max_ = image.min(), image.max()
    logger.debug("New image dtype %s range (%d, %d)", image.dtype, min_, max_)
    return image, {
        "spacing": np.array(spacing),
        "offset": np.array(offset),
        "orientation": orientation,
        "min": min_,
        "max": max_,
    }


# NOTE assumes `data` to be ordered in sitk convention: zyx
def to_sitk(
    data: np.ndarray,
    offset: np.ndarray,
    spacing: np.ndarray,
    savepath: Union[str, Path],
):
    im = sitk.GetImageFromArray(data)
    im.SetOrigin(offset)
    im.SetSpacing(spacing)
    savepath = Path(savepath)
    if not str(savepath).endswith(".mhd"):
        savepath = savepath.with_suffix(".mhd")
    logger.info("Saving scan to '%s'...", savepath)
    sitk.WriteImage(im, savepath, useCompression=True)
    logger.info("DONE")


def load_ASOCA_annotated_centerlines(annotation_fname: str | Path) -> np.ndarray:
    with open(annotation_fname) as fd:
        centerlines = [list(map(float, line.strip().split()[1:])) for line in fd]
    return np.vstack(centerlines if len(centerlines) else [[]])
