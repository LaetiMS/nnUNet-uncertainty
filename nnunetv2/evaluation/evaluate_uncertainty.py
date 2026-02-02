import multiprocessing
import os
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, save_summary_json, compute_metrics_on_folder2, labels_to_list_of_regions
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

import warnings
from pathlib import Path




def compute_uncertainty_metrics(
        reference_file: str,
        uncertainty_file: str,
        image_reader_writer: BaseReaderWriter,
        labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
        ignore_label: int = None,
) -> dict:
    """
    Compute region-conditioned uncertainty statistics.
    """

    # load GT segmentation
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)

    # load uncertainty map (float image!)
    unc, unc_dict = image_reader_writer.read_image(uncertainty_file)
    unc = unc.astype(np.float32)

    # safety first
    unc = np.nan_to_num(unc, nan=0.0, posinf=0.0, neginf=0.0)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results["reference_file"] = reference_file
    results["uncertainty_file"] = uncertainty_file
    results["metrics"] = {}

    for r in labels_or_regions:
        results["metrics"][r] = {}

        mask_ref = region_or_label_to_mask(seg_ref, r)

        if ignore_mask is not None:
            mask_ref = np.logical_and(mask_ref, ~ignore_mask)

        # region statistics
        if mask_ref.any():
            values = unc[mask_ref]
            results["metrics"][r]["mean"] = float(values.mean())
            results["metrics"][r]["std"] = float(values.std())
            results["metrics"][r]["p50"] = float(np.percentile(values, 50))
            results["metrics"][r]["p95"] = float(np.percentile(values, 95))
            results["metrics"][r]["max"] = float(values.max())
            results["metrics"][r]["n_voxels"] = int(mask_ref.sum())
        else:
            # no GT for this region
            results["metrics"][r]["mean"] = np.nan
            results["metrics"][r]["std"] = np.nan
            results["metrics"][r]["p50"] = np.nan
            results["metrics"][r]["p95"] = np.nan
            results["metrics"][r]["max"] = np.nan
            results["metrics"][r]["n_voxels"] = 0

    return results


def compute_uncertainty_metrics_on_folder(
        folder_ref: str,
        folder_pred: str,
        output_file: str,
        image_reader_writer: BaseReaderWriter,
        file_ending: str,
        regions_or_labels,
        ignore_label: int = None,
        uncertainty_names=("variance", "entropy", "normalized_entropy",
                           "mutual_information", "normalized_mutual_information"),
        num_processes: int = default_num_processes,
        chill: bool = True,
):
    if output_file is not None:
        assert output_file.endswith(".json")

    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = [join(folder_ref, f) for f in files_pred]

    uncertainty_dir = join(folder_pred, "uncertainty_maps")

    tasks = []
    for f_ref, f_pred in zip(files_ref, files_pred):
        case_id = f_pred.replace(file_ending, "")
        for unc in uncertainty_names:
            unc_file = join(uncertainty_dir, f"{case_id}_{unc}{file_ending}")
            if not isfile(unc_file):
                if not chill:
                    raise FileNotFoundError(unc_file)
                continue

            tasks.append(
                (
                    join(folder_ref, f_pred),
                    unc_file,
                    image_reader_writer,
                    regions_or_labels,
                    ignore_label,
                    case_id,
                    unc,
                )
            )

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(_compute_uncertainty_case_wrapper, tasks)

    # ---------------- aggregation ----------------
    per_case = {}
    for case_id, unc_name, metrics in results:
        per_case.setdefault(case_id, {})[unc_name] = metrics

    # compute mean over cases
    means = {}
    for unc_name in uncertainty_names:
        means[unc_name] = {}
        for r in regions_or_labels:
            stats = {}
            for k in per_case[next(iter(per_case))][unc_name][r].keys():
                stats[k] = np.nanmean([
                    per_case[c][unc_name][r][k]
                    for c in per_case
                    if unc_name in per_case[c]
                ])
            means[unc_name][r] = stats

    result = {
        "uncertainty_per_case": per_case,
        "mean": means,
    }

    recursive_fix_for_json_export(result)

    if output_file is not None:
        save_summary_json(result, output_file)

    return result

def _compute_uncertainty_case_wrapper(
        reference_file,
        uncertainty_file,
        image_reader_writer,
        regions_or_labels,
        ignore_label,
        case_id,
        unc_name,
):
    metrics = compute_uncertainty_metrics(
        reference_file,
        uncertainty_file,
        image_reader_writer,
        regions_or_labels,
        ignore_label,
    )["metrics"]

    return case_id, unc_name, metrics

def compute_uncertainty_metrics_on_folder2(
        folder_ref: str,
        folder_pred: str,
        dataset_json_file: str,
        plans_file: str,
        output_file: str = None,
        num_processes: int = default_num_processes,
        chill: bool = False,
):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json["file_ending"]

    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    if output_file is None:
        output_file = join(folder_pred, "summary_uncertainty_maps.json")

    lm = PlansManager(plans_file).get_label_manager(dataset_json)

    return compute_uncertainty_metrics_on_folder(
        folder_ref,
        folder_pred,
        output_file,
        rw,
        file_ending,
        lm.foreground_regions if lm.has_regions else lm.foreground_labels,
        lm.ignore_label,
        num_processes=num_processes,
        chill=chill,
    )

def evaluate_uncertainty_folder_entry_point():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty maps produced by nnUNetv2 inference"
    )
    parser.add_argument(
        "gt_folder", type=str,
        help="Folder with ground truth segmentations"
    )
    parser.add_argument(
        "pred_folder", type=str,
        help="Folder with predicted segmentations (must contain uncertainty_maps/)"
    )
    parser.add_argument(
        "-djfile", type=str, required=True,
        help="dataset.json file"
    )
    parser.add_argument(
        "-pfile", type=str, required=True,
        help="plans.json file"
    )
    parser.add_argument(
        "-o", type=str, required=False,
        default=None,
        help="Output file. Optional. Default: pred_folder/summary_uncertainty_maps.json"
    )
    parser.add_argument(
        "-np", type=int, required=False,
        default=default_num_processes,
        help=f"Number of processes used. Optional. Default: {default_num_processes}"
    )
    parser.add_argument(
        "--chill", action="store_true",
        help="Do not crash if uncertainty maps are missing for some cases"
    )

    args = parser.parse_args()

    compute_uncertainty_metrics_on_folder2(
        folder_ref=args.gt_folder,
        folder_pred=args.pred_folder,
        dataset_json_file=args.djfile,
        plans_file=args.pfile,
        output_file=args.o,
        num_processes=args.np,
        chill=args.chill,
    )


if __name__ == '__main__':
    evaluate_uncertainty_folder_entry_point()