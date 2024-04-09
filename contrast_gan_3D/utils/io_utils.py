from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import SimpleITK as sitk
import torchio as tio

from contrast_gan_3D.constants import MAX_HU, MIN_HU, ORIENTATION
from contrast_gan_3D.utils import geometry as geom
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


def load_ASOCA_annotated_centerlines(annotation_fname: Union[str, Path]) -> np.ndarray:
    with open(annotation_fname) as fd:
        centerlines = [list(map(float, line.strip().split()[1:])) for line in fd]
    return np.vstack(centerlines)


def load_centerlines(folder_path: Union[str, Path]) -> np.ndarray:
    folder_path = Path(folder_path)
    vessel_files = folder_path.glob("vessel[0-9]*.txt")
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
    image_path: Union[Path, str], target_orientation: str = ORIENTATION
) -> Tuple[np.ndarray, Dict[str, Union[str, np.ndarray]]]:
    image_path = Path(image_path)
    image = sitk.ReadImage(image_path)

    orientation = get_scan_orientation(image)
    if orientation != target_orientation:
        image = sitk.DICOMOrient(image, target_orientation)
        new_orientation = get_scan_orientation(image)
        logger.info(
            "Changed orientation '%s': %s -> %s",
            str(image_path),
            orientation,
            new_orientation,
        )
        orientation = new_orientation

    spacing, offset = image.GetSpacing(), image.GetOrigin()
    image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)  # DHW -> HWD
    logger.debug(
        "Original image dtype %s range (%s, %s)", image.dtype, image.min(), image.max()
    )

    image = image.astype(np.int16)
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


def sitk_to_h5(
    sitk_img_path: Union[str, Path],
    centerlines: Union[str, Path, np.ndarray],
    ostia: Union[str, Path, np.ndarray],
    h5_output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Path:
    sitk_img_path = Path(sitk_img_path)

    out_dir = sitk_img_path.parent
    if h5_output_dir is not None:
        out_dir = Path(h5_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    image, meta = load_sitk_image(sitk_img_path, **kwargs)  # HWD
    if not isinstance(centerlines, np.ndarray):
        centerlines = load_centerlines(centerlines)
    if not isinstance(ostia, np.ndarray):
        ostia = load_mevis_coords(ostia)[0]

    centerlines_seg = geom.world_to_grid_coords(
        centerlines, meta["offset"], meta["spacing"], image.shape
    )
    # centerline extractor works with WHD oriented images, our convention is HWD
    centerlines_seg = centerlines_seg.transpose(1, 0, 2)  # WHD -> HWD

    logger.debug(
        "CCTA: %s centerlines: %s ostia: %s",
        image.shape,
        centerlines.shape,
        ostia.shape,
    )

    # take care of filenames ending with multiple extensions e.g. .nii.gz
    outpath = (out_dir / Path(stem(sitk_img_path)).with_suffix(".h5")).resolve()
    logger.debug("H5 file: '%s'", str(outpath))

    with h5py.File(outpath, "w") as h5_file:
        group = h5_file.create_group("ccta")

        dset = group.create_dataset("ccta", data=image, compression="lzf")
        dset.attrs.update(meta.items())

        # coronary arteries centerlines and radii, world coordinates
        dset = group.create_dataset("centerlines", data=centerlines, compression="lzf")
        dset.attrs["ostia"] = ostia

        group.create_dataset("centerlines_seg", data=centerlines_seg, compression="lzf")

    return outpath


def hd5_group_attrs_flat(group: h5py.Group) -> dict:
    return {
        attr_name: attr
        for dset in group.values()
        for attr_name, attr in dset.attrs.items()
    }


# NOTE The HD5 file is returned to call .close() once done using it
def load_h5_image(
    image_path: Union[Path, str]
) -> Tuple[h5py.Dataset, Dict[str, np.ndarray], h5py.File]:
    f = h5py.File(image_path)
    group = f["ccta"]
    ccta = group["ccta"]
    return (ccta, hd5_group_attrs_flat(group), f)


def to_itksnap_volume(
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
    sitk.WriteImage(im, savepath, useCompression=True)
