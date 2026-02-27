import multiprocessing
import os
from copy import deepcopy
from typing import Tuple, List, Union
from scipy.ndimage import binary_dilation, binary_erosion

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
        prediction_file: str,
        uncertainty_file: str,
        image_reader_writer: BaseReaderWriter,
        labels_or_regions,
        ignore_label: int = None,
) -> dict:
    """
    Evaluates uncertainty maps by computing distribution statistics:
        - global (all valid voxels)
        - GT-conditioned (inside region r)
        - background (outside region r)

    Uncertainty is NEVER masked by prediction.
    """

    # ---------- Load data ----------
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    seg_ref = seg_ref[0] if seg_ref.ndim == 4 else seg_ref

    seg_pred, _ = image_reader_writer.read_seg(prediction_file)
    seg_pred = seg_pred[0] if seg_pred.ndim == 4 else seg_pred

    unc, _ = image_reader_writer.read_images([uncertainty_file])
    unc = unc.astype(np.float32)
    unc = np.nan_to_num(unc, nan=0.0, posinf=0.0, neginf=0.0)

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
    valid_mask = ~ignore_mask if ignore_mask is not None else np.ones_like(seg_ref, bool)

    results = {
        "reference_file": reference_file,
        "prediction_file": prediction_file,
        "uncertainty_file": uncertainty_file,
        "metrics": {}
    }

    # ---------- GLOBAL ----------
    global_values = unc[:, valid_mask]

    results["metrics"]["global"] = {
        "mean": float(global_values.mean()),
        "std": float(global_values.std()),
        "p50": float(np.percentile(global_values, 50)),
        "p95": float(np.percentile(global_values, 95)),
        "p99": float(np.percentile(global_values, 99)),
        "max": float(global_values.max()),
        "n_voxels": int(global_values.size),
    }

    # ---------- REGION-CONDITIONED ----------
    from scipy.ndimage import binary_dilation, binary_erosion

    # ---------- REGION-CONDITIONED ----------
    for r in labels_or_regions:
        gt_mask = region_or_label_to_mask(seg_ref, r)

        # optional: load prediction if available
        pred_mask = region_or_label_to_mask(seg_pred, r) if seg_pred is not None else np.zeros_like(gt_mask, bool)

        # boundary band (2 voxel shell)
        dilated = binary_dilation(gt_mask, iterations=2)
        eroded = binary_erosion(gt_mask, iterations=2)
        boundary_mask = dilated ^ eroded

        # extended GT + boundary
        gt_boundary_mask = gt_mask | boundary_mask

        # prediction union GT
        pred_union_gt_mask = gt_mask | pred_mask

        # store masks for metrics
        region_masks = {
            "gt": gt_mask,
            "boundary_only": boundary_mask,
            "gt_plus_boundary": gt_boundary_mask,
            "prediction_union_gt": pred_union_gt_mask,
        }

        results["metrics"][r] = {}

        # compute statistics for each mask
        for mask_name, mask in region_masks.items():
            if np.any(mask):
                values = unc[:, mask]
                stats = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(values.max()),
                    "n_voxels": int(mask.sum()),
                }
            else:
                stats = {
                    "mean": np.nan,
                    "std": np.nan,
                    "p50": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "n_voxels": 0,
                }

            results["metrics"][r][mask_name] = stats


        # ---------- BACKGROUND ----------
        bg_mask = (~gt_mask) & valid_mask
        bg_values = unc[:, bg_mask]

        results["metrics"][r]["background"] = {
            "mean": float(bg_values.mean()),
            "std": float(bg_values.std()),
            "p50": float(np.percentile(bg_values, 50)),
            "p95": float(np.percentile(bg_values, 95)),
            "p99": float(np.percentile(bg_values, 99)),
            "max": float(bg_values.max()),
            "n_voxels": int(bg_values.size),
        }

    return results


def _compute_uncertainty_case_wrapper_for_json(
        case_id: str,
        ref_file: str,
        pred_file: str,
        unc_file: str,
        image_reader_writer,
        regions_or_labels,
        ignore_label,
):
    """
    Helper function to compute metrics for one patient + one uncertainty map.
    """
    metrics_dict = compute_uncertainty_metrics(
        reference_file=ref_file,
        prediction_file=pred_file,
        uncertainty_file=unc_file,
        image_reader_writer=image_reader_writer,
        labels_or_regions=regions_or_labels,
        ignore_label=ignore_label,
    )
    metrics_flat = metrics_dict["metrics"]
    recursive_fix_for_json_export(metrics_flat)

    return {
        "reference_file": ref_file,
        "prediction_file" : pred_file,
        "uncertainty_file": unc_file,
        "metrics": metrics_flat
    }


def compute_uncertainty_metrics_on_folder_separate_jsons(
        folder_ref: str,
        folder_pred: str,
        output_dir: str,
        image_reader_writer,
        file_ending: str = ".nii.gz",
        regions_or_labels=None,
        ignore_label: int = None,
        uncertainty_names=(
                "variance",
                "entropy",
                "normalized_entropy",
                "mutual_information",
                "normalized_mutual_information",
        ),
        num_processes: int = default_num_processes,
        chill: bool = True,
):
    """
    Compute uncertainty metrics for all patients and save **one JSON per uncertainty type**.
    """
    folder_ref = Path(folder_ref)
    folder_pred = Path(folder_pred)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    uncertainty_dir = folder_pred / "uncertainty_maps"

    # --- collect reference files ---
    ref_files = list(folder_ref.glob(f"*{file_ending}"))
    ref_map = {f.stem.replace(".nii", ""): f for f in ref_files}


    # --- loop over uncertainty types ---
    for unc_name in uncertainty_names:

        # multiprocessing
        # --- prepare tasks ---
        tasks = []
        for case_id, ref_file in ref_map.items():
            pred_file = folder_pred / f"{case_id}{file_ending}"
            unc_file = uncertainty_dir / f"{case_id}_{unc_name}{file_ending}"
            if not unc_file.is_file():
                if not chill:
                    raise FileNotFoundError(f"{unc_file} not found")
                continue
            tasks.append((case_id, str(ref_file), str(pred_file), str(unc_file), image_reader_writer, regions_or_labels, ignore_label))

        if not tasks:
            print(f"No maps found for uncertainty '{unc_name}', skipping JSON.")
            continue

        # --- compute metrics in parallel ---
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            metric_per_case = pool.starmap(_compute_uncertainty_case_wrapper_for_json, tasks)




        # # no multiprocessing
        # metric_per_case = []
        #
        # for case_id, ref_file in ref_map.items():
        #     unc_file = uncertainty_dir / f"{case_id}_{unc_name}{file_ending}"
        #
        #     if not unc_file.is_file():
        #         if not chill:
        #             raise FileNotFoundError(f"{unc_file} not found")
        #         continue
        #
        #     # compute metrics
        #     metrics_dict = compute_uncertainty_metrics(
        #         reference_file=str(ref_file),
        #         uncertainty_file=str(unc_file),
        #         image_reader_writer=image_reader_writer,
        #         labels_or_regions=regions_or_labels,
        #         ignore_label=ignore_label,
        #     )
        #
        #     metrics_flat = metrics_dict["metrics"]
        #     recursive_fix_for_json_export(metrics_flat)
        #
        #     metric_per_case.append({
        #         "reference_file": str(ref_file),
        #         "uncertainty_file": str(unc_file),
        #         "metrics": metrics_flat
        #     })

        if not metric_per_case:
            print(f"No maps found for uncertainty '{unc_name}', skipping JSON.")
            continue

        # --- compute mean metrics ---
        first_case = metric_per_case[0]["metrics"]
        means = {}

        # global
        means["global"] = {}
        for k in first_case["global"].keys():
            means["global"][k] = float(np.nanmean([c["metrics"]["global"][k] for c in metric_per_case]))

        # per region
        for r in regions_or_labels:
            means[r] = {"gt": {}, "boundary_only": {}, "gt_plus_boundary": {}, "prediction_union_gt": {}, "background": {}}
            for region_type in ("gt", "boundary_only", "prediction", "gt_plus_boundary", "prediction_union_gt", "background"):
                keys = first_case[r][region_type].keys()
                for k in keys:
                    means[r][region_type][k] = float(np.nanmean([c["metrics"][r][region_type][k] for c in metric_per_case]))

        # --- foreground mean (all regions except 0) ---
        fg_regions = [r for r in regions_or_labels if r != 0 and str(r) != "0"]
        foreground_mean = {"gt": {}, "boundary_only": {}, "gt_plus_boundary": {}, "prediction_union_gt": {}, "background": {}}
        for region_type in ("gt", "boundary_only", "gt_plus_boundary", "prediction_union_gt", "background"):
            keys = first_case[fg_regions[0]][region_type].keys()
            for k in keys:
                foreground_mean[region_type][k] = float(
                    np.nanmean([
                        np.nanmean([c["metrics"][r][region_type][k] for r in fg_regions])
                        for c in metric_per_case
                    ])
                )

        # --- global mean ---
        global_mean = {}
        for k in first_case["global"].keys():
            global_mean[k] = float(np.nanmean([c["metrics"]["global"][k] for c in metric_per_case]))

        # --- final result dict ---
        result = {
            "metric_per_case": metric_per_case,
            "mean": means,
            "foreground_mean": foreground_mean,
            "global_mean": global_mean
        }

        # --- save JSON ---
        output_file = output_dir / f"{unc_name}_metrics.json"
        save_summary_json(result, str(output_file))
        print(f"Saved metrics JSON for '{unc_name}' -> {output_file}")



def compute_uncertainty_metrics_on_folder2(
        folder_ref: str,
        folder_pred: str,
        dataset_json_file: str,
        plans_file: str,
        output_dir: str = None,
        num_processes: int = default_num_processes,
        chill: bool = False,
):
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json["file_ending"]

    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # determine output folder
    if output_dir is None:
        output_dir = join(folder_pred, "uncertainty_metrics")
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # get label manager
    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    regions_or_labels = lm.foreground_regions if lm.has_regions else lm.foreground_labels

    # call the new separate JSON function
    return compute_uncertainty_metrics_on_folder_separate_jsons(
        folder_ref=folder_ref,
        folder_pred=folder_pred,
        output_dir=output_dir,
        image_reader_writer=rw,
        file_ending=file_ending,
        regions_or_labels=regions_or_labels,
        ignore_label=lm.ignore_label,
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
        help="Output directory. Optional. Default: pred_folder"
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
        output_dir=args.o,
        num_processes=args.np,
        chill=args.chill,
    )


if __name__ == '__main__':
    evaluate_uncertainty_folder_entry_point()