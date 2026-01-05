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
        variance: [...], voxel-wise variance over classes (on probabilities)
        entropy: [...], predictive entropy
        mutual_information: [...]
    """
    if isinstance(logits_samples, np.ndarray):
        logits_samples = torch.from_numpy(logits_samples)

    # mean over samples (for segmentation)
    mean_logits = logits_samples.mean(dim=0)

    # probabilities: softmax per sample
    probs = torch.softmax(logits_samples, dim=1)
    mean_probs = probs.mean(dim=0)

    # predictive entropy
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=0)

    # variance of probabilities (averaged over classes)
    probs = torch.softmax(logits_samples, dim=1)
    variance = probs.var(dim=0).mean(dim=0) #to avoid computing variance on logits (who are not scale-invariant / highly sensitive to class imbalance)
    # variance over samples, averaged over classes
    #variance = logits_samples.var(dim=0).mean(dim=0) # version where computed on logits

    # add Mutual information here (epistemic uncertainty metric)
    expected_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean(dim=0)
    mutual_information = entropy - expected_entropy

    return mean_logits, variance, entropy, mutual_information


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
      - uncertainty maps (variance + entropy + mutual_information)
    Inspired by resample_and_save and export_prediction_from_logits
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # aggregate
    mean_logits, variance, entropy, mutual_information = aggregate_logits_for_uncertainty(logits_samples)

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
        ["variance", "entropy", "mutual_information"],
        [variance, entropy, mutual_information]
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

    torch.set_num_threads(old_threads)

