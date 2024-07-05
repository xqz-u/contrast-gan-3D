import multiprocessing as mp
from pathlib import Path

from contrast_gan_3D.data import utils as data_u
from contrast_gan_3D.eval.CCTAContrastCorrector import CCTAContrastCorrector
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
