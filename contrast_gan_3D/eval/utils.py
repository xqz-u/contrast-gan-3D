from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from contrast_gan_3D.alias import FoldType, ScanType
from contrast_gan_3D.data.HD5Scan import HD5Scan
from contrast_gan_3D.eval.CCTAContrastCorrector import CCTAContrastCorrector
from contrast_gan_3D.trainer.utils import divide_scans_in_fold
from contrast_gan_3D.utils import geometry as geom


def collect_voxels(
    scan_paths: list[str | Path],
    corrector: CCTAContrastCorrector | None,
    savepath: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    all_ctls, all_corrected_ctls = [[]], [[]]
    all_ostias, all_corrected_ostias = [[]], [[]]
    for scan_path in scan_paths:
        with HD5Scan(scan_path) as scan:
            offset, spacing = scan.meta["offset"], scan.meta["spacing"]
            ccta, centerlines = scan.ccta[::], scan.labelmap[::].astype(bool)
            ostias = geom.world_to_grid_coords(
                scan.centerlines.attrs["ostia"], offset, spacing, ccta.shape
            ).astype(bool)
            if corrector is not None:
                corrected_ccta = corrector(ccta, desc=str(scan_path))
                torch.cuda.empty_cache()  # FIXME why is this needed to avoid errors?!
                if savepath is not None:
                    corrector.save_scan(corrected_ccta, offset, spacing, savepath)
                all_corrected_ctls.append(corrected_ccta[centerlines].numpy())
                all_corrected_ostias.append(corrected_ccta[ostias].numpy())
        all_ctls.append(ccta[centerlines])
        all_ostias.append(ccta[ostias])
    raw = {"centerlines": np.hstack(all_ctls), "ostia": np.hstack(all_ostias)}
    corrected = {
        "centerlines": np.hstack(all_corrected_ctls),
        "ostia": np.hstack(all_corrected_ostias),
    }
    return raw, corrected


def collect_evaluation_histograms(
    evaluation_paths: FoldType, corrector: CCTAContrastCorrector
) -> tuple[
    dict[ScanType, dict[str, np.ndarray]], dict[ScanType, dict[str, np.ndarray]]
]:
    scans_by_label = divide_scans_in_fold(evaluation_paths)
    voxels_by_label, corrected_voxels_by_label = {}, {}
    for st in ScanType:
        eval_paths = scans_by_label[st.value]
        print(st, len(eval_paths))
        voxels, corrected_voxels = collect_voxels(
            eval_paths, None if st == ScanType.OPT else corrector
        )
        voxels_by_label[st], corrected_voxels_by_label[st] = voxels, corrected_voxels
        for k, v in voxels.items():
            print(f"\tTotal voxels {k!r}: {len(v)}")
    return voxels_by_label, corrected_voxels_by_label


def plot_histograms(
    subopt: np.ndarray,
    corrected_subopt: np.ndarray,
    opt: np.ndarray,
    ax: Axes | None = None,
    nbins: int = 80,
    alpha: float = 0.5,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    _, edges0, *_ = ax.hist(
        subopt,
        bins=nbins,
        alpha=alpha,
        density=True,
        label="Suboptimal",
    )
    _, edges1, *_ = ax.hist(
        corrected_subopt,
        bins=nbins,
        alpha=alpha,
        density=True,
        label="Corrected Suboptimal",
    )
    all_edges = np.hstack([edges0, edges1])
    edge_min, edge_max = all_edges.min(), all_edges.max()

    bins, edges = np.histogram(opt, nbins, density=True)
    if edges.min() > edge_min:
        edges, bins = np.insert(edges, 0, edge_min), np.insert(bins, 0, np.zeros(1))
    if edges.max() < edge_max:
        edges, bins = np.append(edges, edge_max), np.append(bins, np.zeros(1))

    X = np.dstack([edges[:-1], edges[1:]]).ravel()
    Y = np.dstack([bins, bins]).ravel()
    ax.plot(X, Y, "k--", label="Optimal")

    return ax
