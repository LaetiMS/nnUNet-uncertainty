# Inference
#  ├── seg_pred (argmax)        → evaluate_prediction (this function)
#  └── mean_probs (softmax)     → evaluate_calibration


import multiprocessing
import numpy as np
from typing import Union, List, Tuple, Optional

from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, save_summary_json, compute_metrics_on_folder2, labels_to_list_of_regions
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def compute_ece(confidence, correct, n_bins=15):
    """
    confidence: [N] predicted confidence in [0, 1] -> max softmax probability per voxel
    correct:    [N] boolean correctness (prediction == GT)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = confidence.shape[0]

    for i in range(n_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if not np.any(mask):
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_probabilistic_metrics(
    reference_file: str,
    probability_file: str,   # <-- NEW (saved softmax)
    image_reader_writer: BaseReaderWriter,
    labels_or_regions,
    ignore_label=None,
    n_bins=15,
    eps: float = 1e-8,
) -> dict:
    """
    This function the compute_metrics equivalent for uncertainty calibration (evaluates the quality of probabilities):
        - loads GT
        - loads probability maps
        - Compute probabilistic / calibration metrics for segmentation::
            - NLL
            - Brier Score
            - ECE
            Metrics are computed voxel-wise and aggregated per case.
            Works for binary and multiclass segmentation
    :param reference_file: Path to ground-truth segmentation file
    :param probability_file: Path to saved softmax probability map file --> todo check shape [C,X,Y,Z] or is it [X,Y,Z] for binary case (foreground probability)
    :param image_reader_writer: Used for consistent loading across nnUNetv2
    :param labels_or_regions: (same object as passed to compute_metrics -->[1] #binary foreground, [(1,2)] #merged regions, [1,2,3] #multiclass)
    :param ignore_label: Label in GT to ignore (e.g. padding, undefined), should match ignore_label used during training
    :param n_bins: Number of confidence bins for ECE (typically 10-20), more bins = noisier, fewer bins -> smoother but less precise
    :param eps: Numerical stability constant -> prevents log(0) in NLL and division-by-zero in ECE bins
    :return:
    Output structure example:
    {
        "reference_file": "...",
        "probability_file": "...",
        "metrics": {
            1: {
                "NLL": 0.42,
                "Brier": 0.13,
                "ECE": 0.06,
                "n_voxels": 123456
            }
        }
    }



    NB (Sanity check shapes at runtime):
    | Variable      | Shape          |
    | ------------- | -------------- |
    | `seg_ref`     | `[X, Y, Z]`    | # after seg_ref = seg_ref[0]
    | `mean_probs`  | `[C, X, Y, Z]` |
    | `p_fg`        | `[X, Y, Z]`    |
    | `ignore_mask` | `[X, Y, Z]`    |

    """

    #############################

    # --- Load GT and probabilities ---
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    seg_ref = seg_ref[0]  # remove singleton --> now shape [X,Y,Z] #todo check
    probs, _ = image_reader_writer.read_seg(probability_file)

    # probs can be [C, X, Y, Z] or [X, Y, Z] (binary FG prob), if binary stored as [X,Y,Z], convert to [2,X,Y,Z]
    if probs.ndim == seg_ref.ndim: #binary case [X,Y,Z]
        probs = np.stack([1.0 - probs, probs], axis=0) # [2,X,Y,Z]

    nr_classes = probs.shape[0]

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {
        "reference_file": reference_file,
        "probability_file": probability_file,
        "metrics": {}
    }

    # --- Loop over labels / regions (same philosophy as compute_metrics) ---
    for r in labels_or_regions:
        results["metrics"][r] = {}

        # GT mask for this region (bool)
        gt_mask = region_or_label_to_mask(seg_ref, r)

        if ignore_mask is not None:
            # add explicit calculation of valid_voxels -> done in compute_tp_fp_fn_tn implicitly (use_mask)
            valid_mask = ~ignore_mask if ignore_mask is not None else np.ones_like(seg_ref, bool)
        else:
            valid_mask = np.ones_like(gt_mask, dtype=bool)

        # only foreground computation by default
        eval_mask = gt_mask & valid_mask #bool

        if not np.any(eval_mask):
            results["metrics"][r]["NLL"] = np.nan
            results["metrics"][r]["Brier"] = np.nan
            results["metrics"][r]["ECE"] = np.nan
            results["metrics"][r]["n_voxels"] = 0
            continue

        # --- Extract probabilities ---
        # Ground truth for selected voxels
        gt_voxels = seg_ref[eval_mask].astype(int) # shape [nr_valid_voxels] or  seg_ref[0][eval_mask] to remove singleton [1,X,Y,Z]
        # probs has shape [C, D, H, W], keep all classes
        probs_eval = probs[:, eval_mask]  # shape [nr_classes, nr_valid_voxels]
        # # transpose to match metric function format
        probs_eval = probs_eval.T  # shape [nr_valid_voxels, nr_classes] #todo check -> check also that there is no issue with shape

        # --- Negative Log-Likelihood (multiclass) ---
        p_true = probs_eval[np.arange(gt_voxels.size), gt_voxels] #if probs_eval was transposed
        # p_true = probs_eval[gt_voxels, np.arange(gt_voxels.size)] # if probs_eval was not transposed
        nll = float(-np.mean(np.log(p_true + eps)))

        # --- Brier score (multiclass) ---
        y_onehot = np.eye(nr_classes)[gt_voxels] # shape [nr_valid_voxels, nr_classes]
        brier = float(np.mean(np.sum((probs_eval - y_onehot) **2, axis=1)))

        # --- Expected Calibration Error (multiclass)---
        pred = np.argmax(probs_eval, axis=1)
        confidence = np.max(probs_eval, axis=1)
        correct = pred == gt_voxels

        ece = compute_ece(confidence, correct, n_bins=n_bins)

        # --- Store results ---
        results["metrics"][r]["NLL"] = nll
        results["metrics"][r]["Brier"] = brier
        results["metrics"][r]["ECE"] = ece
        results["metrics"][r]["n_voxels"] = int(eval_mask.sum())

    return results

def compute_probabilistic_metrics_on_folder(folder_ref: str, folder_prob: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              n_bins: int = 15,
                              eps: float = 1e-8,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
        Folder-level evaluation for probabilistic segmentation metrics:
            - NLL
            - Brier score
            - ECE

        folder_ref  : GT segmentations
        folder_prob : saved probability maps (softmax outputs)
        output_file : must end with .json or be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_prob = subfiles(folder_prob, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_prob, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_prob"
    files_ref = [join(folder_ref, i) for i in files_prob]
    files_prob= [join(folder_prob, i) for i in files_prob]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_prob, [image_reader_writer] * len(files_prob), [regions_or_labels] * len(files_prob), [ignore_label] * len(files_prob))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_probabilistic_metrics,
            list(zip(files_ref, files_prob, [image_reader_writer] * len(files_prob), [regions_or_labels] * len(files_prob),
                     [ignore_label] * len(files_prob), [n_bins] * len(files_prob), [eps] * len(files_prob)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')

def compute_probabilistic_metrics_on_folder2(folder_ref: str, folder_prob: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               n_bins: int = 15,
                               eps: float = 1e-8,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_prob, 'summary_probabilistic.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_probabilistic_metrics_on_folder(folder_ref, folder_prob, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              n_bins, eps, num_processes, chill=chill)


def evaluate_folder_entry_point_probabilistic():
    """
    Computes both summary.json and summary_probabilistic.json
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('prob_folder', type=str, help='folder with softmax probabilities (for calibration metrics)') # added
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-nbins', type=int, required=False, default=15, help='Number of confidence bins for ECE calculation. Default: 15') # added
    parser.add_argument('-eps', type=float, required=False, default=1e-8, help='Epsilon for NLL calculation. Default: 1e-8') # added
    parser.add_argument('-np', type=int, required=False, default=default_num_processes, help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)

    if args.prob_folder is not None:
        # TODO: check path!
        # if i want to save in output folder instead -> replace args.o with output_prob
        # output_prob = (
        #     join(args.prob_folder, 'summary_probabilistic.json')
        #     if args.o is None
        #     else args.o.replace('.json', '_probabilistic.json')
        # )
        compute_probabilistic_metrics_on_folder2(args.gt_folder, args.prob_folder, args.djfile, args.pfile, args.o, args.nbins, args.eps, args.np, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/<uncertainty_method>'
    folder_prob =  '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/<uncertainty_method>'
    #output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary_probabilities.json'

    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    nr_ece_bins = 15
    eps = 1e-8
    num_processes = 12

    # prob_results = compute_probabilistic_metrics(
    #     reference_file=gt_path,
    #     probability_file=softmax_path,
    #     image_reader_writer=image_reader_writer,
    #     labels_or_regions=[1],
    #     ignore_label=0,
    #     nr_bins = nr_ece_bins,
    #     eps = eps
    # )
    compute_probabilistic_metrics_on_folder(folder_ref, folder_prob, output_file, image_reader_writer, file_ending, regions,
                              ignore_label,nr_ece_bins, eps, num_processes)



