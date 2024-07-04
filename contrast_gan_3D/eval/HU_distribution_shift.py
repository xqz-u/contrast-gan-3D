from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    load_pickle,
    save_pickle,
)

from contrast_gan_3D.alias import ArrayShape, FoldType, ScanType
from contrast_gan_3D.eval.CCTAContrastCorrector import CCTAContrastCorrector
from contrast_gan_3D.eval.utils import collect_voxels, read_myocardium_seg_path
from contrast_gan_3D.trainer.utils import divide_scans_in_fold
from contrast_gan_3D.utils import visualization as viz


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


def eval_HU_distribution_shift(
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
