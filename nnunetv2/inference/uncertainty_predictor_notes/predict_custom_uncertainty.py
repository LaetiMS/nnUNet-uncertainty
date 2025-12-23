# Initialize an nnUNet predictor
# Inject desired behaviors:
    # TTA (extended augmentations)
    # MC‑Dropout sampling
    # SWAG sampling from saved snapshots
    # Layer ensemble from deep supervised logits
# # Aggregate results to compute:
    # Mean prediction
    # Uncertainty map


# Standard Deterministic Prediction
# 1. Initialize the predictor once with the trained model, then:

predictor.initialize_from_trained_model_folder(
    model_folder,
    use_folds=(0,),  # or multiple folds
    checkpoint_name="checkpoint_final.pth"
)
predictor.predict_from_files(...)

# 2. Test‑Time Augmentation (TTA)
# nnUNetv2 already supports mirroring TTA internally.
# To add more augmentations (rotation, intensity, etc.):
# Wrap the nnUNet predictor’s predict_from_npy_array call inside your own augment‑deaugment loop:

for aug in augmentations:
    aug_input = apply_augmentation(raw_input, aug)
    logits = predictor.predict_from_npy_array(aug_input)
    deaug_logits = undo_augmentation(logits, aug)
    logits_list.append(deaug_logits)
mean_logits = np.mean(logits_list, axis=0)


# 3. MC‑Dropout
# You want dropout active at inference. To do this (You do this outside nnUNet’s inference driver but call into its sliding window logic.):
# - Modify your network architecture class to keep dropout on when evaluating.
# - Wrap prediction in a loop with dropout activated :

predictor.network.train()
for _ in range(N):
    logits = predictor.predict_from_npy_array(input)
    all_logits.append(logits)
mean_logits = np.mean(all_logits, axis=0)
uncertainty_map = compute_entropy(all_logits)


# 4. SWAG Sampling
# SWAG needs multiple snapshots you saved during training (in swag_snapshots/). You can:
# - Load each snapshot into the same network architecture.
# - For each snapshot, do a sliding‑window prediction:

for ckpt in checkpoint_paths:
    predictor.load_checkpoint(ckpt)
    logits = predictor.predict_from_npy_array(input)
    all_logits.append(logits)

# Aggregate outputs & compute uncertainty.
# This is very similar to MC‑Dropout but with weights sampled from SWAG.

# 5. Layer Ensemble
# If your model returns multiple outputs (e.g., deep supervision outputs):
#     - Modify your forward pass to return all logits heads
#     - Collect them per head:

all_layer_logits = predictor.predict_layerwise(input)
# mean across layers, uncertainty maps, etc.

# This needs custom logic in your model subclass, which is already how you plan to extract layered outputs.

# 🛠 Files You Don’t Need to Modify
# - predict_from_raw_data.py: core predictor logic
# - export_prediction.py: output conversion logic with resampling and shape handling
# These can remain intact. Your custom code uses them rather than replacing them.


# higher-level skeleton:

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np

def predict_with_custom_uncertainty(model_folder, img_paths, out_folder,
                                    method="mc_dropout", n_samples=20, augmentations=None):
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))

    # Load raw inputs once
    raw_inputs = load_nifti_images(img_paths)

    if method == "det":
        predictor.predict_from_files(img_paths, out_folder)

    elif method == "tta":
        logits_list = []
        for aug in augmentations:
            aug_imgs = apply_aug(raw_inputs, aug)
            logits = predictor.predict_from_npy_array(aug_imgs)
            logits_list.append(invert_aug(logits, aug))
        mean_logits = np.mean(logits_list, axis=0)

    elif method == "mc_dropout":
        predictor.network.train()
        logits_list = []
        for i in range(n_samples):
            logits = predictor.predict_from_npy_array(raw_inputs)
            logits_list.append(logits)
        mean_logits = np.mean(logits_list, axis=0)

    elif method == "swag":
        logits_list = []
        for ckpt in sorted_checkpoint_paths:
            predictor.load_checkpoint(ckpt)
            logits = predictor.predict_from_npy_array(raw_inputs)
            logits_list.append(logits)
        mean_logits = np.mean(logits_list, axis=0)

    # Export final mean prediction + uncertainty
    save_mean_and_uncert(out_folder, mean_logits, logits_list)
