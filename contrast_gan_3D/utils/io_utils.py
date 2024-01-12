from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import SimpleITK as sitk

from contrast_gan_3D.utils.logging_utils import create_logger

logger = create_logger(name=__name__)


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
    image_path: Union[Path, str], is_cadrads: bool = False
) -> Tuple[h5py.Dataset, Dict[str, np.ndarray]]:
    content = h5py.File(image_path)
    image_ds = content["ccta"]["ccta"]
    centerlines = (
        np.array(image_ds.attrs["ctl_points"])
        if is_cadrads
        else content["ccta"]["centerlines"]
    )
    return image_ds, {
        "spacing": np.array(image_ds.attrs["spacing"]),
        "offset": np.array(image_ds.attrs["offset"]),
        "centerlines": centerlines,
    }


def load_centerlines(
    folder_path: Union[str, Path], create: bool = True
) -> np.ndarray:
    folder_path = Path(folder_path)

    vessel_files = folder_path.glob("vessel[0-9]*.txt")
    centerlines = [np.loadtxt(v) for v in vessel_files]
    centerlines = np.concatenate(centerlines or [[]], axis=0, dtype=np.float32)

    if create and len(centerlines):
        savepath = folder_path / "centerlines.npz"
        np.savez_compressed(savepath, centerlines=centerlines)
        logger.debug("Saved centerlines to '%s' with key 'centerlines'", str(savepath))

    return centerlines


def sitk_to_h5(
    sitk_img_path: Union[str, Path],
    centerlines: Union[str, Path, np.ndarray],
    h5_output_dir: Optional[Union[str, Path]] = None,
):
    sitk_img_path = Path(sitk_img_path)
    assert str(sitk_img_path).endswith(".mhd") or str(sitk_img_path).endswith(
        ".nii.gz"
    ), f"Usupported file extension for {str(sitk_img_path)!r}"

    out_dir = sitk_img_path.parent
    if h5_output_dir is not None:
        out_dir = Path(h5_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    image, meta = load_sitk_image(sitk_img_path)

    if not isinstance(centerlines, np.ndarray):
        centerlines = load_centerlines(centerlines)

    logger.debug("CCTA: %s centerlines: %s", image.shape, centerlines.shape)

    # take care of filenames ending with multiple extensions, e.g. .nii.gz
    basename = f"{str(sitk_img_path).split('/')[-1].split('.')[0]}"
    outpath = (out_dir / f"{basename}.h5").resolve()
    logger.debug("H5 file: '%s'", str(outpath))

    with h5py.File(outpath, "w") as h5_file:
        group = h5_file.create_group("ccta")

        group.create_dataset("centerlines", data=centerlines)
        dset = group.create_dataset("ccta", data=image, compression="lzf")

        dset.attrs["spacing"] = meta["spacing"]
        dset.attrs["offset"] = meta["offset"]
