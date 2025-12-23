# Using the custom trainer modifications, where snapshots are saved every N epochs in fold_0/swag_snapshots/, we can write a modular predictor wrapper that:
#
# Here we :
# - Load each snapshot.
# - Perform sliding-window prediction on your test images using nnUNet’s predictor logic.
# - Aggregates logits to compute both mean prediction and uncertainty maps.
#
# NB: We’ll avoid modifying predict_from_raw_data.py or export_prediction.py

import os
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import subfiles

def predict_swag(
    model_folder: str,
    img_dir: str,
    out_dir: str,
    fold: int = 0,
    device: str = "cuda",
):
    """
    Predict using SWAG snapshots saved during training.
    model_folder: path to nnUNet results / fold_X
    img_dir: folder with imagesTs
    out_dir: where to save seg.nii.gz, uncertainty.nii.gz, meta.json
    """

    # Setup output folder
    os.makedirs(out_dir, exist_ok=True)

    # Path to SWAG snapshots
    swag_dir = os.path.join(model_folder, "swag_snapshots")
    if not os.path.exists(swag_dir):
        raise FileNotFoundError(f"SWAG snapshot folder not found: {swag_dir}")

    # Get list of snapshot checkpoints
    snapshots = sorted(subfiles(swag_dir, suffix=".pth", join=True))
    if len(snapshots) == 0:
        raise ValueError("No SWAG snapshots found!")

    print(f"[SWAG] Found {len(snapshots)} snapshots. Using them for prediction...")

    # Initialize nnUNet predictor with a dummy checkpoint (will reload each snapshot)
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,))

    # Collect all logits
    all_logits = []

    for ckpt in snapshots:
        print(f"[SWAG] Loading snapshot: {ckpt}")
        predictor.load_checkpoint(ckpt)  # replace network weights
        logits = predictor.predict_from_folder(img_dir)  # sliding-window, returns np.array per image
        all_logits.append(logits)

    # Convert list of logits to numpy array: (num_snapshots, num_images, H, W, D, C)
    all_logits = np.stack(all_logits, axis=0)

    # Compute mean and uncertainty
    mean_logits = np.mean(all_logits, axis=0)
    # Uncertainty: predictive entropy per voxel
    prob = torch.softmax(torch.tensor(mean_logits), dim=-1).numpy()
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-8, 1.0)), axis=-1)

    # Save outputs (for each image)
    predictor.save_prediction_mean_and_uncertainty(out_dir, mean_logits, entropy)

    print(f"[SWAG] Predictions saved to: {out_dir}")
