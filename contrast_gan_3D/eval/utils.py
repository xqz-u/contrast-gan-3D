import multiprocessing as mp
from pathlib import Path

import numpy as np

from contrast_gan_3D.data import utils as data_u
from contrast_gan_3D.eval.CCTAContrastCorrector import CCTAContrastCorrector
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils, set_multiprocessing_start_method


def correct_patient(
    corrector: CCTAContrastCorrector,
    savedir: str | Path,
    patient_path: str | Path,
    batch_size: int = 16,
):
    scan, meta = data_u.load_patient(str(patient_path))
    offset, spacing = meta["offset"], meta["spacing"]
    corrected_ccta = corrector(
        scan[..., 0], batch_size=batch_size, desc=str(patient_path)
    )
    savepath = Path(savedir) / io_utils.stem(str(patient_path))
    corrector.save_scan(corrected_ccta, offset, spacing, savepath)


def parallel_correct_patients(
    corrector: CCTAContrastCorrector,
    savedir: str | Path,
    patient_paths: list[str | Path],
    batch_size: int = 16,
    processes: int = 4,
):
    set_multiprocessing_start_method("spawn")
    with mp.Pool(processes) as pool:
        pool.starmap(
            correct_patient,
            [(corrector, savedir, p, batch_size) for p in patient_paths],
        )


def collect_voxels(
    scan_paths: list[str | Path],
    myocardium_paths: list[str | Path],
    corrector: CCTAContrastCorrector | None,
    savedir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    all_ctls, all_corrected_ctls = [[]], [[]]
    all_ostias, all_corrected_ostias = [[]], [[]]
    all_myocardia, all_corrected_myocardia = [[]], [[]]

    if savedir is not None:
        savedir = Path(savedir)
        savedir.mkdir(exist_ok=True, parents=True)

    for scan_path, myocardium_seg_path in zip(scan_paths, myocardium_paths):
        myocardium_seg, _ = io_utils.load_sitk_image(
            myocardium_seg_path, segmentation=True
        )
        myocardium_seg = myocardium_seg.astype(bool)

        scan, meta = data_u.load_patient(str(scan_path))
        offset, spacing = meta["offset"], meta["spacing"]
        ccta, centerlines = scan[..., 0], scan[..., 1].astype(bool)

        ostias = geom.world_to_grid_coords(
            meta["ostia_world"], offset, spacing, scan.shape[:-1]
        ).astype(bool)

        indexers = [centerlines, ostias, myocardium_seg]

        if corrector is not None:
            corrected_ccta = corrector(ccta, desc=str(scan_path))
            if savedir is not None:
                savepath = savedir / io_utils.stem(str(scan_path))
                corrector.save_scan(corrected_ccta, offset, spacing, savepath)

            for cont, indx in zip(
                [all_corrected_ctls, all_corrected_ostias, all_corrected_myocardia],
                indexers,
            ):
                cont.append(corrected_ccta[indx].numpy())
        for cont, indx in zip([all_ctls, all_ostias, all_myocardia], indexers):
            cont.append(ccta[indx])

    keys = ["centerlines", "ostia", "myocardium"]
    raw = {k: np.hstack(v) for k, v in zip(keys, [all_ctls, all_ostias, all_myocardia])}
    corrected = {
        k: np.hstack(v)
        for k, v in zip(
            keys, [all_corrected_ctls, all_corrected_ostias, all_corrected_myocardia]
        )
    }
    return raw, corrected


def read_myocardium_seg_path(preproc_ccta_path: str) -> Path:
    myocardium_seg_path = preproc_ccta_path.replace("preproc", "segmentation") + ".mhd"
    myocardium_seg_path = Path(myocardium_seg_path)
    if myocardium_seg_path.is_symlink():
        myocardium_seg_path = myocardium_seg_path.readlink()
    assert myocardium_seg_path.is_file()
    return myocardium_seg_path


