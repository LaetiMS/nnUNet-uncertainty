import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Tuple, Union, List, Optional
from time import sleep


import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from torch import nn
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.export_uncertainty_prediction import export_uncertainty_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, _getDefaultValue
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


from glob import glob

from contextlib import contextmanager





def set_network_mode_for_inference(network: torch.nn.Module, enable_mc_dropout: bool):
    """
    Sets the network in inference mode with optional MC dropout.

    Args:
        network: torch.nn.Module, the network to set
        enable_mc_dropout: if True, dropout layers stay active during inference
    """
    if enable_mc_dropout:
        network.train()  # enable dropout at inference
        # Force BatchNorm layers to eval (do not update running stats)
        for m in network.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    else:
        network.eval()

def strip_orig_mod_prefix(state_dict):
    return {
        k.replace('_orig_mod.', '', 1): v
        for k, v in state_dict.items()
    }



class MCDropoutPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        super().__init__(tile_step_size,use_gaussian,use_mirroring,perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        self.enable_mc_dropout = True
    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        # added for dropout
        if self.enable_mc_dropout:
            for m in self.network.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.train()


        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

class TTAextendedPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 noise_variance=(0.0, 0.03), # added
                 gamma_range=(0.9, 1.1) # added
                 ):
        super().__init__(tile_step_size,use_gaussian,use_mirroring,perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)
        self.enable_TTA_extended = True
        # default --> disable TTA --> self.use_mirroring = not args.disable_tta

        # stochastic, image-only TTAs
        self.noise_tf = GaussianNoiseTransform(
            noise_variance=noise_variance,
            p_per_channel=1.0,
            synchronize_channels=True
        )

        self.gamma_tf = GammaTransform(
            gamma=gamma_range,
            p_invert_image=0,
            synchronize_channels=True,
            p_per_channel=1.0,
            p_retain_stats=True
        )


    def _apply_stochastic_tta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply weak stochastic augmentations.
        Expects x as torch.Tensor [B, C, ...]
        """
        data = {"image": x}

        # sample independently each call
        if torch.rand(1).item() < 0.5:
            data = self.noise_tf(data)
        if torch.rand(1).item() < 0.5:
            data = self.gamma_tf(data)

        return data["image"]

    #todo: check whether i need to remove : torch.inference_mode here too?
    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        # 1. original prediction
        prediction = super()._internal_maybe_mirror_and_predict(x)

        # 2. stochastic TTA (use batchgeneratorsv2)

        if self.enable_TTA_extended:
            # maybe add separate TTA extendors --> we can run multiple and see how well each augmentation explains uncertainties
            xt = self._apply_stochastic_tta(x)
            # call original nnUNet mirroring logic
            prediction = super()._internal_maybe_mirror_and_predict(xt)

        return prediction


class UncertaintyPredictor(nnUNetPredictor):
    """
    NB: when predicting on test set BLANK is used, when predicting the final validation manual_initalization is used.
    Notice that self.list_of_parameters is None in that case as self.network in the trainer is already a nn.Module.
    When initializing the predictor for the test prediction, the model is loaded from a checkpoint containing weights.
    Hence, why in predict_logits_from_preprocessed_data they are unpacked see; self.network.load_state_dict!
    See also function perform_actual_validation in Trainer to see how the predictor is initialized.

    overwrites every nnUNetPredictor function calling export_prediction_from_logits with export_uncertainty_from_logits
    - predict_from_files
    -- predict_from_data_iterator
    --- predict_logits_from_preprocessed_data -> predict_logits_from_preprocessed_data_uncertainty
        ---- predict_sliding_window_return_logits
    --- export_prediction_from_logits -> export_uncertainty_from_logits
        ---- convert_predicted_logits_to_segmentation_with_correct_shape (unchanged)
    --- convert_predicted_logits_to_segmentation_with_correct_shape (unchanged)
    (- predict_single_npy_array) -> never used irrelevant
        -- predict_logits_from_preprocessed_data
        -- export_prediction_from_logits
        -- convert_predicted_logits_to_segmentation_with_correct_shape
    (- predict_from_list_of_npy_arrays --> no change needed as change is in predict_from_data_iterator)
    """
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 enable_mc_dropout: bool = False,
                 enable_swag_prediction: bool = False,
                 enable_tta_nnunet_limits: bool = False,
                 enable_tta_agressive: bool = False,
                 enable_tta_paper: bool = False,
                 ):
        super().__init__(tile_step_size,use_gaussian,use_mirroring,perform_everything_on_device, device, verbose, verbose_preprocessing, allow_tqdm)

        # which uncertainty method should be added
        self.enable_mc_dropout = enable_mc_dropout
        self.enable_swag_predict = enable_swag_prediction
        self.enable_tta_nnunet_limits = enable_tta_nnunet_limits
        self.enable_tta_agressive = enable_tta_agressive
        self.enable_tta_paper = enable_tta_paper

        self.enable_TTA_extended = True if self.enable_tta_nnunet_limits or self.enable_tta_agressive or self.enable_tta_paper else False
        self.uncertainty_method_is_in_use = True if self.enable_TTA_extended or self.enable_mc_dropout or self.enable_swag_predict else False
        # todo: add layered_ensembles (both in __init__ and _get_uncertainty_method_name) + implement method
        self.uncertainty_method_name = self._get_uncertainty_method_name()

    def _get_uncertainty_method_name(self) -> str:
        """
        Aggregates the different uncertainty methods that are used in the predictor
        """
        name = ''
        if self.enable_mc_dropout:
            name += f"{('_' if not name =='' else '')}mc_dropout"
        if self.enable_swag_predict:
            name += f"{('_' if not name =='' else '')}swag"
        if self.use_mirroring or self.enable_TTA_extended:
            name += f"{('_' if not name =='' else '')}tta"
            if self.use_mirroring:
                name += '_mirroring'
            if self.enable_tta_nnunet_limits:
                name += '_limits'
            if self.enable_tta_agressive:
                name += '_agressive'
            if self.enable_tta_paper:
                name += '_paper'
        if name == '':
            name = 'no_uncertainty'
        return name

        # todo: add layered_ensembles

    @contextmanager
    def _inference_context(self):
        """
        Allows to select the inference mode that certain functions will be run with
        """
        if self.enable_mc_dropout:
            with torch.no_grad():
                yield
        else:
            with torch.inference_mode():
                yield

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model.
        - addition: the weights of all swag_snapshots/checkpoints are added to the list_of_parameters. Currently the checkpoint_final.pth is crucial. 1. Because my training didnt save the final checkpoint in swag_snapshots (cause i did 800 to 1000, instead of 799 to 999).
            Note it exploits how the weights of the different folds would be added!
        """
        if self.enable_swag_predict:
            if checkpoint_name == 'checkpoint_final.pth':
                if use_folds is None:
                    use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

                if isinstance(use_folds, str):
                    use_folds = [use_folds]

                if len(use_folds) != 0 and use_folds[0] != 'all':
                    f = int(use_folds[0])
                    parameters = []
                    fold_dir = join(model_training_output_dir, f'fold_{f}')

                    # --- load SWAG snapshots ---
                    swag_snapshots_path = join(fold_dir, 'swag_snapshots')
                    # check if path exists
                    if isdir(swag_snapshots_path):
                        swag_files = sorted(glob(join(swag_snapshots_path, 'epoch_*.pth')))
                        if len(swag_files) == 0:
                            raise RuntimeError(f"No SWAG snapshots found in {swag_snapshots_path}")
                        else:

                            swag_checkpoints = []
                            for swag_ckpt in swag_files:
                                #your SWAG snapshots are raw state_dicts, not nnU-Net–style checkpoints
                                sd = torch.load(swag_ckpt, map_location=torch.device('cpu'), weights_only=False)
                                # remove _orig_mod from all keys -> allows to load_state_dict()
                                sd = strip_orig_mod_prefix(sd)
                                swag_checkpoints.append(sd)

                            parameters.extend(swag_checkpoints)


                    # --- load final checkpoint ---
                    final_ckpt = torch.load(join(fold_dir, checkpoint_name), map_location=torch.device('cpu'), weights_only=False)
                    parameters.append(final_ckpt['network_weights'])

                    trainer_name = final_ckpt['trainer_name']
                    configuration_name = final_ckpt['init_args']['configuration']
                    inference_allowed_mirroring_axes = final_ckpt['inference_allowed_mirroring_axes'] if \
                        'inference_allowed_mirroring_axes' in final_ckpt.keys() else None

                    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
                    plans = load_json(join(model_training_output_dir, 'plans.json'))
                    plans_manager = PlansManager(plans)

                    configuration_manager = plans_manager.get_configuration(configuration_name)
                    # restore network
                    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
                    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
                    if trainer_class is None:
                        raise RuntimeError(
                            f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                            f'Please place it there (in any .py file)!')
                    network = trainer_class.build_network_architecture(
                        configuration_manager.network_arch_class_name,
                        configuration_manager.network_arch_init_kwargs,
                        configuration_manager.network_arch_init_kwargs_req_import,
                        num_input_channels,
                        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                        enable_deep_supervision=False
                    )

                    self.plans_manager = plans_manager
                    self.configuration_manager = configuration_manager
                    self.list_of_parameters = parameters

                    # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
                    network.load_state_dict(parameters[0])

                    self.network = network

                    self.dataset_json = dataset_json
                    self.trainer_name = trainer_name
                    self.allowed_mirroring_axes = inference_allowed_mirroring_axes
                    self.label_manager = plans_manager.get_label_manager(dataset_json)
                    if ('nnUNet_compile' in os.environ.keys()) and (
                            os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                            and not isinstance(self.network, OptimizedModule):
                        print('Using torch.compile')
                        self.network = torch.compile(self.network)
                else:
                    RaiseNotImplementedError('swag predictions can only be run one fold at the time, it is currently not implemented for multiple folds')

            else:
                # todo: fix this issue .... by retraining my models
                RaiseNotImplementedError(
                    'swag predictions currently requires the checkpoint by default to be checkpoint_final (this is because in my runs the last checkpoint was not recorded, ... oups)')

        else:
            super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)

    def predict_from_data_iterator_new_ofile(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        copy of predict_from_data_iterator, but the output file path is modified to account for the uncertainty estimation_method (i.e. no_uncertainty or mirror_only).
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                # added start
                ofile_base = preprocessed['ofile']
                # structure based on uncertainty method
                if ofile_base is not None:
                    case_id = os.path.basename(ofile_base)
                    base_dir = os.path.dirname(ofile_base)
                    ofile = os.path.join(base_dir, "uncertainty", str(self.uncertainty_method_name), case_id)
                    os.makedirs(os.path.dirname(ofile), exist_ok=True)
                    print(f'\nPredicting {os.path.basename(ofile_base)}:')
                else:
                    ofile = None
                    print(f'\nPredicting image of shape {data.shape}:')
                # added end


                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret



    def predict_from_data_iterator_uncertainty(
            self,
            data_iterator,
            save_probabilities: bool = False,
            num_processes_segmentation_export: int = default_num_processes
    ):
        """
        Uncertainty-aware prediction.
        Each element returned by data_iterator must be a dict with keys:
            - 'data'
            - 'ofile'
            - 'data_properties'
        If 'ofile' is None, the result will be returned instead of written to a file

        This function:
          - runs SWAG / MC Dropout / TTA inference
          - aggregates uncertainty on the main process
          - exports segmentation + uncertainty maps


        NB: Example of resulting structure (i.e. if self.uncertainty_method_name is mc_dropout, swag, deep_ensemble etc.).
        PS: Make sure to run evaluation pipeline once per method pointing to probabilities/<method_name>/:
        predictions/
        └── uncertainty/
            ├── mc_dropout/
            │   ├── case_001.nii.gz
            │   ├── case_001_entropy.nii.gz
            │   ├── case_001_variance.nii.gz
            │   ├── case_001_mutual_information.nii.gz
            │   └── case_001_normalized_entropy.nii.gz
            │
            ├── deep_ensemble/
            │   └── ...
            │
            └── swag/
                └── ...

        """

        results = [] # r
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list= [i for i in export_pool._pool]
            r = []

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile_base = preprocessed['ofile']
                # structure based on uncertainty method
                if ofile_base is not None:
                    case_id = os.path.basename(ofile_base)
                    base_dir = os.path.dirname(ofile_base)
                    ofile = os.path.join(base_dir, "uncertainty", str(self.uncertainty_method_name), case_id)
                    os.makedirs(os.path.dirname(ofile), exist_ok=True)
                    print(f'\nPredicting {os.path.basename(ofile_base)}:')
                else:
                    ofile = None
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # on main GPU
                # Run stochastic interference and convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors

                logits_samples = []

                # Use inference_mode when possible (lower memory), otherwise no_grad for MC Dropout
                with self._inference_context():
                    # predict_logits_from_preprocessed_data_with_uncertainty
                    # returns a tensor of shape [S, C, H, W, D] (already stacked)
                    logits_samples = self.predict_logits_from_preprocessed_data_with_uncertainty(data)

                # At this point:
                # - GPU memory pressure already happened (inside sliding window)
                # - logits_samples may be on GPU or CPU depending on OOM fallback

                # Ensure results are on CPU
                if isinstance(logits_samples, torch.Tensor) and logits_samples.device.type == "cuda":
                    logits_samples = logits_samples.cpu()

                # Convert to numpy only if required downstream
                logits_samples = logits_samples.numpy()



                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)


                # 2. Export (main process!) and offload CPU postprocessing
                if ofile is not None:
                    r.append(
                        export_pool.starmap_async(
                            export_uncertainty_from_logits,
                            [(
                                logits_samples,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                ofile,
                                save_probabilities
                            )]
                        )
                    )

                    print(f'done with {os.path.basename(ofile)}')
                else:
                    # return results instead of writing
                    results.append(logits_samples)
                print('done')

                # Wait for all exports to finish
                for job in r:
                    job.get()

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # cleanup (nnU-Net style)
        compute_gaussian.cache_clear()
        empty_cache(self.device)

        return results



    def predict_logits_from_preprocessed_data_with_uncertainty(
            self,
            data: torch.Tensor,
            mc_passes: int = 10,
            tta_passes: int = 4,
    ) -> torch.Tensor:
        """
        Returns:
            logits_samples: [S, C, ...], where S is the number of stochastic forward passes
            where S = (#SWAG checkpoints) × mc_passes × tta_passes

        Memory-efficient: moves each pass to CPU immediately (to avoid OOM on GPU)
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        logits_samples = []

        for params in self.list_of_parameters:  # e.g., the weights of the SWAG snapshots + checkpoint_final.pth
            # load state dict
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # network.eval or network.train
            set_network_mode_for_inference(self.network, self.enable_mc_dropout)

            mc_iters = mc_passes if self.enable_mc_dropout else 1
            tta_iters = tta_passes if self.enable_TTA_extended else 1

            for _ in range(mc_iters):
                for _ in range(tta_iters):
                    # sliding window forward
                    #todo: add actual tta stuff here -> see TTApredictor, currently there is no TTaugmentation added to data

                    # select appropriate inference context (inference_mode or no_grad)
                    with self._inference_context():
                        logits = self.predict_sliding_window_return_logits_uncertainty(data)

                    # logits is already on CPU if GPU OOM hapened
                    # If it survived on GPU, move it once
                    if logits.device.type == "cuda":
                        logits = logits.cpu()
                    logits_samples.append(logits)

        # stack all passes on CPU
        logits_samples = torch.stack(logits_samples, dim=0)  # shape [S, C, ...]

        torch.set_num_threads(n_threads)
        return logits_samples

    def _internal_predict_sliding_window_return_logits_uncertainty(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        """
        copy of _internal_predict_sliding_window_return_logits
        changes:
        - uses _internal_maybe_mirror_and_predict_uncertainty (which is _internal_maybe_mirror_and_predict but without @torch.inference_mode decorator)
        - nolonger decorated with @torch.inference_mode() as needs to allow for network.eval for MCDropout (i.e. @torch.no_grad()).

        NB: using torch.no_grad() instead of torch.inference_mode() will cause GPU OOM and will shift the prediction onto CPU
        """

        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    with self._inference_context():
                        prediction = self._internal_maybe_mirror_and_predict_uncertainty(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits


    def predict_sliding_window_return_logits_uncertainty(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        """
        only modification is replace self.network.eval() with set_network_mode_for_inference
        and uses _internal_predict_sliding_window_return_logits_uncertainty as _internal_predict_sliding_window_return_logits
        
        Compared to og - it is nolonger decorated with @torch.inference_mode() as needs to allow for network.eval for MCDropout (i.e. @torch.no_grad()). 
        NB: using torch.no_grad() instead of torch.inference_mode() will cause GPU OOM and will shift the prediction onto CPU
        """
        self.network = self.network.to(self.device)

        # network.eval or network.train
        set_network_mode_for_inference(self.network, self.enable_mc_dropout)

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                       'constant', {'value': 0}, True,
                                                       None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    # select appropriate inference context (inference_mode or no_grad)
                    with self._inference_context():
                        predicted_logits = self._internal_predict_sliding_window_return_logits_uncertainty(data, slicers,
                                                                                           self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    # select appropriate inference context (inference_mode or no_grad)
                    with self._inference_context():
                        predicted_logits = self._internal_predict_sliding_window_return_logits_uncertainty(data, slicers, False)
            else:
                # select appropriate inference context (inference_mode or no_grad)
                with self._inference_context():
                    predicted_logits = self._internal_predict_sliding_window_return_logits_uncertainty(data, slicers,
                                                                                       self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def _internal_maybe_mirror_and_predict_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        copy of _internal_maybe_mirror_and_predict but without decorator
        instead call with self.inference_context()
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def predict_from_files_uncertainty(
            # mirror of predict_from_files
            self,
            list_of_lists_or_source_folder:
            Union[str, List[List[str]]],
            output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
            save_probabilities: bool = False,
            overwrite: bool = True,
            num_processes_preprocessing: int = default_num_processes,
            num_processes_segmentation_export: int = default_num_processes,
            folder_with_segs_from_prev_stage: str = None,
            num_parts: int = 1,
            part_id: int = 0
    ):
        """
        Uncertainty-aware prediction entry point.
        Mirrors predict_from_files, but uses uncertainty inference + export.
        Only difference to predict_from_files -> final return calls uncertainty equivalent method!
        """

        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)
        if not self.uncertainty_method_is_in_use:
            # has correct output path, but does not calculate uncertainty maps
            return self.predict_from_data_iterator_new_ofile(data_iterator, save_probabilities=save_probabilities)
        else:
            return self.predict_from_data_iterator_uncertainty(data_iterator, save_probabilities=save_probabilities)
        # todo: A few very important details -> currently regarless of whether we want the uncertainty_prediction or not it will go through the uncertainty predictor.



def predict_entry_point_uncertainty():
    # adapted from predict_entry_point() in predict_from_raw_data.py
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=_getDefaultValue('nnUNet_npp', int, 3),
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=_getDefaultValue('nnUNet_nps', int, 3),
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    # new
    parser.add_argument('--activate_mc_dropout_prediction', action='store_true', required=False, default=False,
                        help='Set this flag to activate dropout during the prediction.')
    parser.add_argument('--activate_swag_predict', action='store_true', required=False, default=False,
                        help='Set this flag to predict for each checkpoint saved in swag_snapshots. ')
    parser.add_argument('--activate_tta_nnunet_limits', action='store_true', required=False, default=False,
                        help='Set this flag to activate an extended TTA - training augmentations but more extreme - during the prediction.')
    parser.add_argument('--activate_tta_agressive', action='store_true', required=False, default=False,
                        help='Set this flag to activate an extended TTA - agressive augmentations from the torchio library that were not used in training - during the prediction.')
    parser.add_argument('--activate_tta_paper', action='store_true', required=False, default=False,
                        help='Set this flag to activate an extended TTA - augmentations used in paper: https://arxiv.org/abs/1807.07356 - during the prediction.')
    parser.add_argument('--activate_layered_ensembles', action='store_true', required=False, default=False,
                        help='Set this flag to extract the layers to derive uncertainties for an'
                             'approximate layered ensembles uncertainty estimation.')
    # todo: add --activate_deep_ensembles here? (can be similar to the use of swag_predict only that instead we store all checkpoints in the same DE location? --> decide still

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = UncertaintyPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                enable_mc_dropout=args.activate_mc_dropout_prediction, # added
                                enable_swag_prediction = args.activate_swag_predict,
                                enable_tta_nnunet_limits = args.activate_tta_nnunet_limits,
                                enable_tta_agressive = args.activate_tta_agressive,
                                enable_tta_paper = args.activate_tta_paper,
                                #enable_layered_ensembles=args.activate_layered_ensembles,
                                )

    # def __init__(self,
    #              tile_step_size: float = 0.5,
    #              use_gaussian: bool = True,
    #              use_mirroring: bool = True,
    #              perform_everything_on_device: bool = True,
    #              device: torch.device = torch.device('cuda'),
    #              verbose: bool = False,
    #              verbose_preprocessing: bool = False,
    #              allow_tqdm: bool = True,
    #              enable_mc_dropout: bool = False,
    #              enable_swag_prediction: bool = False,
    #              enable_tta_nnunet_limits: bool = False,
    #              enable_tta_agressive: bool = False,
    #              enable_tta_paper: bool = False,
    #              ):
    #todo: here
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    run_sequential = args.nps == 0 and args.npp == 0

    if run_sequential:

        print("Running in non-multiprocessing mode")
        predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                                overwrite=not args.continue_prediction,
                                                folder_with_segs_from_prev_stage=args.prev_stage_predictions)

    else:

        # predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
        #                              overwrite=not args.continue_prediction,
        #                              num_processes_preprocessing=args.npp,
        #                              num_processes_segmentation_export=args.nps,
        #                              folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        #                              num_parts=args.num_parts,
        #                              part_id=args.part_id)
        predictor.predict_from_files_uncertainty(args.i, args.o, save_probabilities=args.save_probabilities,
                                                 overwrite=not args.continue_prediction,
                                                 num_processes_preprocessing=args.npp,
                                                 num_processes_segmentation_export=args.nps,
                                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                                 num_parts=args.num_parts,
                                                 part_id=args.part_id)



    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           device=device)
    #                           part_id=args.part_id,

# if __name__ == '__main__':
#     ########################## predict a bunch of files
#     from nnunetv2.paths import nnUNet_results, nnUNet_raw
#
#     predictor = UncertaintyPredictor(
#         tile_step_size=0.5,
#         use_gaussian=True,
#         use_mirroring=True,
#         perform_everything_on_device=True,
#         device=torch.device('cuda', 0),
#         verbose=False,
#         verbose_preprocessing=False,
#         allow_tqdm=True
#     )
#     # predictor.initialize_from_trained_model_folder(
#     #     join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),
#     #     use_folds=(0,),
#     #     checkpoint_name='checkpoint_final.pth',
#     # )
#     predictor.initialize_from_trained_model_folder(
#         join(nnUNet_results, r'Dataset001_unilateral_axial_only\nnUNetTrainer__nnUNetPlans__3d_fullres'),
#         use_folds=(0,),
#         checkpoint_name='checkpoint_final.pth',
#     )
#     # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
#     #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
#     #                              save_probabilities=False, overwrite=False,
#     #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
#     #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
#     #
#     # # predict a numpy array
#     # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
#     #
#     # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
#     # ret = predictor.predict_single_npy_array(img, props, None, None, False)
#     #
#     # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
#     # ret = predictor.predict_from_data_iterator(iterator, False, 1)
#
#     # ret = predictor.predict_from_files_sequential(
#     #     [['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz'], ['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_005_0000.nii.gz']],
#     #     '/home/isensee/temp/tmp', False, True, None
#     # )
#     ret = predictor.predict_from_files_sequential(
#         [[r'C:\Users\Laetitia\Documents\Programming\PdMLaetitia\data_nnUnet_compatible\raw\Dataset001_unilateral_axial_only\imagesTs\DUKE_005_A_0000.nii.gz'], [r'C:\Users\Laetitia\Documents\Programming\PdMLaetitia\data_nnUnet_compatible\raw\Dataset001_unilateral_axial_only\imagesTs\DUKE_021_B_0000.nii.gz']],
#         r'C:\Users\Laetitia\Documents\Programming\PdMLaetitia\data_nnUnet_compatible\nnUNet_uncertainty\Dataset001_unilateral_axial_only', False, True, None
#     )
#
#     #todo: modify predict_entry_point(_modelfolder) to incorporate new predictor -> here you can add new arguments also!
# adapted from run_training.py
if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    #os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # multiprocessing.set_start_method("spawn")
    predict_entry_point_uncertainty()


    # todo: add self.uncertainty_method_name to predictor
    # todo: add a if enable_method -> self.uncertainty_method name