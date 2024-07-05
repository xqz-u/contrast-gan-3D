import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np

from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.trainer import utils as train_u
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils


def collect_patient_voxels(
    scan_path: str | Path,
    centerline_path: str | Path,
    myocardium_path: str | Path,
) -> dict[str, np.ndarray]:
    ccta, meta = io_utils.load_sitk_image(scan_path)
    myocardium_mask, _ = io_utils.load_sitk_image(myocardium_path, segmentation=True)

    centerlines_world = io_utils.load_centerlines(centerline_path)[..., :3]
    ostia_world, _ = io_utils.load_mevis_coords(Path(centerline_path) / "ostia.xml")

    offset, spacing = meta["offset"], meta["spacing"]
    ostia_mask = geom.world_to_grid_coords(ostia_world, offset, spacing, ccta.shape)
    centerlines_mask = geom.world_to_grid_coords(
        centerlines_world, offset, spacing, ccta.shape
    )

    indexers = [m.astype(bool) for m in [centerlines_mask, ostia_mask, myocardium_mask]]
    return {
        k: ccta[idx] for k, idx in zip(["centerlines", "ostia", "myocardium"], indexers)
    }


def _helper(*args: tuple[list[str | Path], int]) -> tuple[int, dict[str, np.ndarray]]:
    args, label = args
    return (label, collect_patient_voxels(*args))


def _aggregate_voxels(
    r: list[tuple[int, dict[str, np.ndarray]]]
) -> dict[ScanType, dict[str, np.ndarray]]:
    ret = defaultdict(lambda: defaultdict(list))
    for lab, d in r:
        for tag, v in d.items():
            ret[ScanType(lab)][tag].append(v)
    return {
        lab: {tag: np.concatenate(v) for tag, v in d.items()} for lab, d in ret.items()
    }


def collect_voxels_intensity(
    evaluation_paths: list[tuple[list[str | Path], int]], processes: int = 8
) -> dict[ScanType, dict[str, np.ndarray]]:
    print("Scans distribution by label:")
    pprint(
        {
            ScanType(k): len(v)
            for k, v in train_u.divide_scans_in_fold(evaluation_paths).items()
        }
    )

    with mp.Pool(processes=processes) as pool:
        res = pool.starmap(_helper, evaluation_paths)
    voxels = _aggregate_voxels(res)

    for k, d in voxels.items():
        print(k)
        for kk, v in d.items():
            print(f"\tTotal voxels {kk!r}: {len(v)}")

    return voxels
