import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from contrast_gan_3D.alias import ScanType
from contrast_gan_3D.utils import geometry as geom
from contrast_gan_3D.utils import io_utils


def read_ASOCA_annotations(patient_dir: Path) -> dict[str, np.ndarray]:
    return {"centerlines": io_utils.load_ASOCA_annotated_centerlines(patient_dir)}


def read_IDR_CADRADS_annotations(patient_dir: Path) -> dict[str, np.ndarray]:
    # 3 annotated arteries, 4 annotations each artery, expected shape: (3, 4, 3)
    ret = {}
    for art in ["LAD", "LCX", "RCA"]:
        annot_fname = patient_dir / f"{art}.txt"
        if not annot_fname.is_file():
            print(f"Skip missing annotation {str(annot_fname)!r}")
            continue
        art_annotation = np.loadtxt(annot_fname)
        if len(art_annotation) != 4:
            print(f"{str(annot_fname)!r} has only {len(art_annotation)} annotations")
        ret[art] = art_annotation
    return ret


def marker_recall_rate(distance_to_marker: np.ndarray, threshold: float = 5.0) -> float:
    return (distance_to_marker <= threshold).sum() / len(distance_to_marker)


def find_closest_centerlines_to_annatations(
    annotations_dir_path: str | Path,
    centerlines_dir_path: str | Path,
    annot_read_fn: Callable[
        [Path], dict[str, np.ndarray]
    ] = read_IDR_CADRADS_annotations,
    verbose: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    if verbose:
        print("Annotations:", str(annotations_dir_path))
        print("Centerlines:", str(centerlines_dir_path))
        print("Centerlines reader:", annot_read_fn)
    centerlines = io_utils.load_centerlines(centerlines_dir_path)[..., :3]

    annotation_coords_named = annot_read_fn(Path(annotations_dir_path))
    artery_dist_dict = {}
    for name, annot_coord in annotation_coords_named.items():
        if not annot_coord.size:
            print(f"Missing annonations for {str(annotations_dir_path)!r}")
            continue
        ctls_euclidean_dist = geom.pointwise_euclidean_distance(
            centerlines, annot_coord
        )
        # save the distance of the points closest to each annotation
        idx, val = ctls_euclidean_dist.argmin(0), ctls_euclidean_dist.min(0)
        artery_dist_dict[name] = {"z_idx": idx, "dist": val}
    return artery_dist_dict


def _helper(*args):
    (label, *args), kwargs = args
    return (label, find_closest_centerlines_to_annatations(*args, **kwargs))


def _parallel_marker_recall_rate(
    annotations_root_dir: str | Path,
    centerlines_root_dir: str | Path,
    labels_df: pd.DataFrame,
    processes: int = 8,
    **kwargs,
) -> list[tuple[int, dict[str, dict[str, np.ndarray]]]]:
    args = [
        ((lab, ap[0], cp[0]), kwargs)
        for lab, name in labels_df[["label", "ID"]].values
        if len((ap := list(Path(annotations_root_dir).glob(f"*{name}*"))))
        and len((cp := list(Path(centerlines_root_dir).glob(f"*{name}*"))))
    ]
    with mp.Pool(processes=processes) as pool:
        return pool.starmap(_helper, args)


def _aggregate_mrr(r: dict) -> tuple[dict, dict]:
    # structure: [(label, {tag: {z_idx: [], dist: []}}), ...]
    # output: {label: {tag: {z_idx: np.ndarray, dist: np.ndarray}}, ...}
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metrics = defaultdict(dict)
    for label, inner_dict in r:
        for tag_name, distance_dict in inner_dict.items():
            for k, v in distance_dict.items():
                result[label][tag_name][k].append(v)

    for label, tag_dict in result.items():
        for tag_name, distance_dict in tag_dict.items():
            for k, v in distance_dict.items():
                result[label][tag_name][k] = np.concatenate(v)
            metrics[ScanType(label)][tag_name] = marker_recall_rate(
                result[label][tag_name]["dist"]
            )
    result = {
        ScanType(k): {ik: dict(iv) for ik, iv in v.items()} for k, v in result.items()
    }
    return result, dict(metrics)


def eval_model_marker_recall_rate(
    centerlines_root_dir: Path | str,
    annotations_root_dir: Path | str,
    labels_df: pd.DataFrame,
    **kwargs,
) -> tuple[
    dict[ScanType, dict[str, dict[str, np.ndarray]]],
    dict[ScanType, dict[str, np.ndarray]],
]:
    return _aggregate_mrr(
        _parallel_marker_recall_rate(
            annotations_root_dir, centerlines_root_dir, labels_df, **kwargs
        )
    )


def summarize_marker_recall_rate(
    distances: dict[ScanType, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    aggregated, subopt = {"optimal": {}}, defaultdict(list)
    for scan_type, dd in distances.items():
        for annot_tag, ddd in dd.items():
            if scan_type in {ScanType.LOW, ScanType.HIGH}:
                subopt[annot_tag].append(ddd["dist"])
            else:
                aggregated["optimal"][annot_tag] = marker_recall_rate(ddd["dist"])
    aggregated["suboptimal"] = {
        art_t: marker_recall_rate(np.concatenate(v)) for art_t, v in subopt.items()
    }
    return aggregated
