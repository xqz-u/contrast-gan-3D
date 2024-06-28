from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    load_pickle,
    save_pickle,
)

from contrast_gan_3D.alias import ArrayShape, FoldType, ScanType
from contrast_gan_3D.data import utils as data_u
from contrast_gan_3D.eval.CCTAContrastCorrector import CCTAContrastCorrector
from contrast_gan_3D.trainer.utils import divide_scans_in_fold
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils
from contrast_gan_3D.utils import visualization as viz


def correct_patients(
    corrector: CCTAContrastCorrector,
    patient_paths: list[str | Path],
    savedir: str | Path,
    batch_size: int = 16,
):
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for p in patient_paths:
        scan, meta = data_u.load_patient(str(p))
        offset, spacing = meta["offset"], meta["spacing"]
        corrected_ccta = corrector(scan[..., 0], batch_size=batch_size, desc=str(p))
        savepath = savedir / io_utils.stem(str(p))
        corrector.save_scan(corrected_ccta, offset, spacing, savepath)


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
            # torch.cuda.empty_cache()  # FIXME why is this needed to avoid ever growing memory?
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


def collect_evaluation_histograms(
    evaluation_paths: FoldType,
    corrector: CCTAContrastCorrector | None = None,
    itk_export_dir: str | Path | None = None,
) -> tuple[
    dict[ScanType, dict[str, np.ndarray]], dict[ScanType, dict[str, np.ndarray]]
]:
    scans_by_label = divide_scans_in_fold(evaluation_paths)
    print("Scans distribution by label:")
    pprint({ScanType(k): len(v) for k, v in scans_by_label.items()})

    voxels_by_label, corrected_voxels_by_label = {}, {}
    for st in ScanType:
        eval_paths, myocardium_paths = list(zip(*scans_by_label[st.value]))
        print(st)
        voxels, corrected_voxels = collect_voxels(
            eval_paths,
            myocardium_paths,
            None if st == ScanType.OPT else corrector,
            savedir=itk_export_dir,
        )
        voxels_by_label[st], corrected_voxels_by_label[st] = voxels, corrected_voxels
        for k, v in voxels.items():
            print(f"\tTotal voxels {k!r}: {len(v)}")
    return voxels_by_label, corrected_voxels_by_label


def read_myocardium_seg_path(preproc_ccta_path: str) -> Path:
    myocardium_seg_path = preproc_ccta_path.replace("preproc", "segmentation") + ".mhd"
    myocardium_seg_path = Path(myocardium_seg_path)
    if myocardium_seg_path.is_symlink():
        myocardium_seg_path = myocardium_seg_path.readlink()
    assert myocardium_seg_path.is_file()
    return myocardium_seg_path


def evaluate_one_model(
    model_path: str | Path,
    ccta_eval_paths: list[tuple[str | Path, int]],
    inference_patch_size: ArrayShape,
    voxels_savepath: str | Path,
    plot_savepath: str | Path | None,
    device: torch.device,
    itk_export_dir: str | Path | None = None,
    show: bool = True,
) -> tuple[dict[ScanType, dict[str, np.ndarray]], ...]:
    voxels_savepath = Path(voxels_savepath).with_suffix(".pkl")
    if voxels_savepath.is_file():
        eval_voxels = load_pickle(voxels_savepath)
        og_voxels, corrected_voxels = eval_voxels["raw"], eval_voxels["corrected"]
        print(f"Loaded evaluation voxels from {str(voxels_savepath)!r}")

        print("Scans distribution by label:")
        pprint(
            {
                ScanType(k): len(v)
                for k, v in divide_scans_in_fold(ccta_eval_paths).items()
            }
        )
        for st, voxels in og_voxels.items():
            print(st)
            for k, v in voxels.items():
                print(f"\tTotal voxels {k!r}: {len(v)}")
    else:
        scan_and_myoc_paths = [
            ([scan_path, read_myocardium_seg_path(scan_path)], label)
            for scan_path, label in ccta_eval_paths
        ]
        corrector = CCTAContrastCorrector.from_checkpoint(
            inference_patch_size, device, checkpoint_path=model_path
        )
        og_voxels, corrected_voxels = collect_evaluation_histograms(
            scan_and_myoc_paths, corrector=corrector, itk_export_dir=itk_export_dir
        )
        save_pickle({"raw": og_voxels, "corrected": corrected_voxels}, voxels_savepath)
        print(f"Saved evaluation voxels to {str(voxels_savepath)!r}")

    viz.create_eval_plot(og_voxels, corrected_voxels, plot_savepath, show)

    return og_voxels, corrected_voxels
