# Inference
#  ├── seg_pred (argmax)        → evaluate_prediction (this function)
#  └── mean_probs (softmax)     → evaluate_calibration

import numpy as np
from typing import Union, List, Tuple, Optional
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask

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
    prob_file: str,   # <-- NEW (saved softmax)
    image_reader_writer: BaseReaderWriter,
    labels_or_regions,
    ignore_label=None,
    n_bins=15,
    eps: float = 1e-8,
) -> dict:
    """
    This function the compute_metrics equivalent for uncertainty calibration:
        - loads GT
        - loads probability maps
        - Compute probabilistic / calibration metrics for segmentation::
            - NLL
            - Brier Score
            - ECE
            Metrics are computed voxel-wise and aggregated per case.
            Works for binary and multiclass segmentation
    :param reference_file: Path to ground-truth segmentation file
    :param prob_file: Path to saved softmax probability map file --> todo check shape [C,X,Y,Z] or is it [X,Y,Z] for binary case (foreground probability)
    :param image_reader_writer: Used for consistent loading across nnUNetv2
    :param labels_or_regions: (same object as passed to compute_metrics -->[1] #binary foreground, [(1,2)] #merged regions, [1,2,3] #multiclass)
    :param ignore_label: Label in GT to ignore (e.g. padding, undefined), should match ignore_label used during training
    :param n_bins: Number of confidense bins for ECE (typically 10-20), more bins = noisier, fewer bins -> smoother but less precise
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
    | `seg_ref`     | `[X, Y, Z]`    |
    | `mean_probs`  | `[C, X, Y, Z]` |
    | `p_fg`        | `[X, Y, Z]`    |
    | `ignore_mask` | `[X, Y, Z]`    |

    """

    #############################
    # TODO: ADAPT FOR MULTICLASS CURRENTLY ONLY BINARY!!!

    # --- Load GT and probabilities ---
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    seg_ref = seg_ref[0]  # remove singleton --> now shape [X,Y,Z] #todo check
    probs, _ = image_reader_writer.read_seg(prob_file)

    # probs can be [C, X, Y, Z] or [X, Y, Z] (binary FG prob), if binary stored as [X,Y,Z], convert to [2,X,Y,Z]
    if probs.ndim == seg_ref.ndim: #binary case [X,Y,Z]
        probs = np.stack([1.0 - probs, probs], axis=0) # [2,X,Y,Z]

    nr_classes = probs.shape[0]

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {
        "reference_file": reference_file,
        "probability_file": prob_file,
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

if __name__ == '__main__':
    prob_results = compute_probabilistic_metrics(
        reference_file=gt_path,
        probability_file=softmax_path,
        image_reader_writer=rw,
        labels_or_regions=[1],
        ignore_label=0,
        foreground_only=True,
    )
