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
    foreground_only: bool = True,
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
    :param foreground_only: Whether to computer metrics only on foreground voxels (common in medical segmentation calibration)
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

    # mean_probs, prob_dict = image_reader_writer.read_prob(probability_file)
    #
    # # NLL binary case, per region
    # p = mean_probs[fg_class]  # foreground probability
    # y = (seg_ref == r).astype(np.float32)
    #
    # nll = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # nll = nll[~ignore_mask].mean()
    #
    # # Brier score
    # brier = ((p - y) ** 2)
    # brier = brier[~ignore_mask].mean()
    #
    # # ECE (voxel-wise)
    # conf = np.maximum(p, 1 - p)
    # pred = (p >= 0.5)
    # correct = (pred == y)
    #
    # ece = compute_ece(conf, correct, n_bins)

    #############################
    # TODO: ADAPT FOR MULTICLASS CURRENTLY ONLY BINARY!!!

    # --- Load GT and probabilities ---
    seg_ref, _ = image_reader_writer.read_seg(reference_file)
    probs, _ = image_reader_writer.read_seg(prob_file)

    # probs can be [C, X, Y, Z] or [X, Y, Z] (binary FG prob), if binary stored as [X,Y,Z], convert to [2,X,Y,Z]
    if probs.ndim == seg_ref.ndim:
        # binary case: probs = foreground probability
        probs = np.stack([1.0 - probs, probs], axis=0)

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

        # GT mask for this region
        gt_mask = region_or_label_to_mask(seg_ref, r)

        if ignore_mask is not None:
            # add explicit calculation of valid_voxels -> done in compute_tp_fp_fn_tn implicitly (use_mask)
            valid_mask = ~ignore_mask if ignore_mask is not None else np.ones_like(seg_ref, bool)
        else:
            valid_mask = np.ones_like(gt_mask, dtype=bool)

        # only foreground computation by default
        eval_mask = gt_mask & valid_mask

        if not np.any(eval_mask):
            results["metrics"][r]["NLL"] = np.nan
            results["metrics"][r]["Brier"] = np.nan
            results["metrics"][r]["ECE"] = np.nan
            results["metrics"][r]["n_voxels"] = 0
            continue

        # --- Extract probabilities ---
        # Binary foreground probability
        p_fg = probs[1] if nr_classes == 2 else probs[r]

        y = gt_mask.astype(np.float32)

        p = p_fg[eval_mask]
        y = y[eval_mask]

        # --- Negative Log-Likelihood ---
        nll = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        #p_true = probs_eval[gt, np.arange(gt.size)]
        #nll = -np.log(p_true + eps)
        nll = float(nll.mean())

        # --- Brier score ---
        brier = float(np.mean((p - y) ** 2))

        # --- Expected Calibration Error ---
        confidence = np.maximum(p, 1.0 - p)
        prediction = p >= 0.5
        correct = prediction == y.astype(bool)

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
