from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

def aggregate_logits_for_uncertainty(
    logits_samples: Union[torch.Tensor, np.ndarray]
):
    """
    logits_samples: [S, C, ...]
    Returns:
        mean_logits: [C, ...]
        variance: [...], voxel-wise variance over classes
        entropy: [...], predictive entropy
    """
    if isinstance(logits_samples, np.ndarray):
        logits_samples = torch.from_numpy(logits_samples)

    # mean over samples
    mean_logits = logits_samples.mean(dim=0)

    # softmax per sample
    probs = torch.softmax(logits_samples, dim=1)
    mean_probs = probs.mean(dim=0)

    # predictive entropy
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=0)

    # variance over samples, averaged over classes
    variance = logits_samples.var(dim=0).mean(dim=0)

    return mean_logits, variance, entropy

def export_uncertainty_from_logits(
    logits_samples: Union[torch.Tensor, np.ndarray],
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str],
    output_file_truncated: str,
    save_probabilities: bool = False,
    num_threads_torch: int = default_num_processes
):
    """
    Saves:
      - segmentation (from mean prediction)
      - uncertainty maps (variance + entropy)
    """

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # aggregate
    mean_logits, variance, entropy = aggregate_logits_for_uncertainty(logits_samples)

    # --- export segmentation (reuse existing code!) ---
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)

    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        mean_logits,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
        num_threads_torch=num_threads_torch
    )
    del mean_logits

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret


    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(
        segmentation_final,
        output_file_truncated + dataset_json_dict_or_file['file_ending'],
        properties_dict
    )

    # --- export uncertainty maps ---
    # resample uncertainty to original space (same as probabilities)
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    for name, vol in zip(
        ["variance", "entropy"],
        [variance, entropy]
    ):
        vol = configuration_manager.resampling_fn_probabilities(
            vol[None],  # fake channel dim
            properties_dict['shape_after_cropping_and_before_resampling'],
            current_spacing,
            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
        )[0]

        vol = insert_crop_into_image(
            np.zeros(properties_dict['shape_before_cropping'], dtype=np.float32),
            vol.cpu().numpy(),
            properties_dict['bbox_used_for_cropping']
        )

        vol = vol.transpose(plans_manager.transpose_backward)

        rw.write_seg(
            vol,
            output_file_truncated + f"_{name}" + dataset_json_dict_or_file['file_ending'],
            properties_dict
        )
