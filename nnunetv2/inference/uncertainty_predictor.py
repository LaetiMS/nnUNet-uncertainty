#import os
from typing import Tuple, Union, List, Optional


import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch import nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import subfiles
from nnunetv2.utilities.helpers import empty_cache, dummy_context




class UncertaintyPredictor(nnUNetPredictor):
    """
    Extends nnUNetPredictor to support multiple uncertainty estimation methods:
    - SWAG
    - MC-Dropout (added to predict_sliding_window_return_logits and __init__)
    - TTA (test-time augmentation)
    - Layered ensembles (deep supervision)
    """
    def __init__(self, *args, enable_mc_dropout=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self. enable_mc_dropout = enable_mc_dropout
        # init file from nnUNetPredictor creates self.trainer_name, self.plans_manager, self.dataset_json etc.
        # def __init__(self,
        #              tile_step_size: float = 0.5,
        #              use_gaussian: bool = True,
        #              use_mirroring: bool = True,
        #              perform_everything_on_device: bool = True,
        #              device: torch.device = torch.device('cuda'),
        #              verbose: bool = False,
        #              verbose_preprocessing: bool = False,
        #              allow_tqdm: bool = True):
        #     self.verbose = verbose
        #     self.verbose_preprocessing = verbose_preprocessing
        #     self.allow_tqdm = allow_tqdm
        #
        #     self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        #         self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None
        #
        #     self.tile_step_size = tile_step_size
        #     self.use_gaussian = use_gaussian
        #     self.use_mirroring = use_mirroring
        #     if device.type == 'cuda':
        #         torch.backends.cudnn.benchmark = True
        #     else:
        #         print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
        #         perform_everything_on_device = False
        #     self.device = device
        #     self.perform_everything_on_device = perform_everything_on_device

    # def __init__(self, model_folder, folds=(0,), device="cuda"):
    #     super().__init__()
        self.model_folder = model_folder
        self.folds = folds
        self.device = device
        self.initialize_from_trained_model_folder(model_folder, use_folds=folds)

    # def initialize_from_trained_model_folder(self, model_training_output_dir: str,
    #                                          use_folds: Union[Tuple[Union[int, str]], None],
    #                                          checkpoint_name: str = 'checkpoint_final.pth'):
    #     """
    #     This is used when making predictions with a trained model
    #     """
    #     if use_folds is None:
    #         use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)
    #
    #     dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    #     plans = load_json(join(model_training_output_dir, 'plans.json'))
    #     plans_manager = PlansManager(plans)
    #
    #     if isinstance(use_folds, str):
    #         use_folds = [use_folds]
    #
    #     parameters = []
    #     for i, f in enumerate(use_folds):
    #         f = int(f) if f != 'all' else f
    #         checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
    #                                 map_location=torch.device('cpu'), weights_only=False)
    #         if i == 0:
    #             trainer_name = checkpoint['trainer_name']
    #             configuration_name = checkpoint['init_args']['configuration']
    #             inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
    #                 'inference_allowed_mirroring_axes' in checkpoint.keys() else None
    #
    #         parameters.append(checkpoint['network_weights'])
    #
    #     configuration_manager = plans_manager.get_configuration(configuration_name)
    #     # restore network
    #     num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    #     trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
    #                                                 trainer_name, 'nnunetv2.training.nnUNetTrainer')
    #     if trainer_class is None:
    #         raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
    #                            f'Please place it there (in any .py file)!')
    #     network = trainer_class.build_network_architecture(
    #         configuration_manager.network_arch_class_name,
    #         configuration_manager.network_arch_init_kwargs,
    #         configuration_manager.network_arch_init_kwargs_req_import,
    #         num_input_channels,
    #         plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
    #         enable_deep_supervision=False
    #     )
    #
    #     self.plans_manager = plans_manager
    #     self.configuration_manager = configuration_manager
    #     self.list_of_parameters = parameters
    #
    #     # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
    #     network.load_state_dict(parameters[0])
    #
    #     self.network = network
    #
    #     self.dataset_json = dataset_json
    #     self.trainer_name = trainer_name
    #     self.allowed_mirroring_axes = inference_allowed_mirroring_axes
    #     self.label_manager = plans_manager.get_label_manager(dataset_json)
    #     if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
    #             and not isinstance(self.network, OptimizedModule):
    #         print('Using torch.compile')
    #         self.network = torch.compile(self.network)
    #
    # def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
    #                           configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
    #                           dataset_json: dict, trainer_name: str,
    #                           inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
    #     """
    #     This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
    #     """
    #     self.plans_manager = plans_manager
    #     self.configuration_manager = configuration_manager
    #     self.list_of_parameters = parameters
    #     self.network = network
    #     self.dataset_json = dataset_json
    #     self.trainer_name = trainer_name
    #     self.allowed_mirroring_axes = inference_allowed_mirroring_axes
    #     self.label_manager = plans_manager.get_label_manager(dataset_json)
    #     allow_compile = True
    #     allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
    #                 os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
    #     allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
    #     if isinstance(self.network, DistributedDataParallel):
    #         allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
    #     if allow_compile:
    #         print('Using torch.compile')
    #         self.network = torch.compile(self.network)

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

    def predict_swag(self, img_dir, out_dir):
        #initialize_from_trained_model_folder -> check checkpoint!
        swag_dir = os.path.join(self.model_folder, "swag_snapshots")
        snapshots = sorted(subfiles(swag_dir, suffix=".pth", join=True))

        if len(snapshots) == 0:
            raise FileNotFoundError(f"No SWAG snapshots found in {swag_dir}")

        all_logits = []
        for ckpt in snapshots:
            print(f"[SWAG] Loading snapshot: {ckpt}")
            self.load_checkpoint(ckpt)
            logits = self.predict_from_folder(img_dir)
            all_logits.append(logits)

        self._save_mean_and_uncertainty(all_logits, out_dir)

    def predict_mc_dropout(self, img_dir, out_dir, n_samples=20):
        self.network.train()  # enable dropout
        all_logits = []

        for i in range(n_samples):
            print(f"[MC-Dropout] Sample {i+1}/{n_samples}")
            logits = self.predict_from_folder(img_dir)
            all_logits.append(logits)

        self._save_mean_and_uncertainty(all_logits, out_dir)
        self.network.eval()  # reset

    def predict_tta(self, img_dir, out_dir, augmentations):
        """
        augmentations: list of callable functions that apply/deapply augmentations
        """
        all_logits = []

        for aug_fn in augmentations:
            aug_imgs = aug_fn(img_dir)  # apply augmentation
            logits = self.predict_from_folder(aug_imgs)
            deaug_logits = aug_fn(aug_imgs, reverse=True)  # de-augment
            all_logits.append(deaug_logits)

        self._save_mean_and_uncertainty(all_logits, out_dir)

    def predict_layer_ensemble(self, img_dir, out_dir, last_n_layers=3):
        """
        Requires network.forward() to return a list of logits (deep supervision outputs)
        """
        all_logits = []
        for _ in range(last_n_layers):
            logits = self.predict_from_folder(img_dir, return_all_layers=True)
            # only keep the last n layers
            all_logits.append(logits[-last_n_layers:])

        self._save_mean_and_uncertainty(all_logits, out_dir)

    def _save_mean_and_uncertainty(self, logits_list, out_dir):
        """
        Aggregates multiple logits and saves mean prediction + uncertainty map
        """
        os.makedirs(out_dir, exist_ok=True)
        all_logits = np.stack(logits_list, axis=0)  # (num_samples, num_images, H, W, D, C)
        mean_logits = np.mean(all_logits, axis=0)

        # Predictive entropy as uncertainty
        prob = torch.softmax(torch.tensor(mean_logits), dim=-1).numpy()
        entropy = -np.sum(prob * np.log(np.clip(prob, 1e-8, 1.0)), axis=-1)

        # Save final segmentation
        self.save_prediction_mean_and_uncertainty(out_dir, mean_logits, entropy)
        print(f"[UncertaintyPredictor] Saved mean prediction and uncertainty to {out_dir}")
