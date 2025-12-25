import os
from typing import Tuple, Union, List, Optional


import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from torch import nn
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, _getDefaultValue
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.file_path_utilities import get_output_folder

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
        self.enable_mc_dropout = enable_mc_dropout
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




def predict_entry_point_custom():
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
    parser.add_argument('--activate_dropout_prediction', action='store_true', required=False, default=False,
                        help='Set this flag to activate dropout during the prediction.')
    parser.add_argument('--activate_extended_TTA', action='store_true', required=False, default=False,
                        help='Set this flag to activate extended TTA during the prediction.')
    parser.add_argument('--activate_layered_ensembles', action='store_true', required=False, default=False,
                        help='Set this flag to extract the layers to derive uncertainties for an'
                             'approximate layered ensembles uncertainty estimation.')


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
                                enable_mc_dropout=args.activate_dropout_prediction, # added
                                )
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

        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
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
    predict_entry_point_custom()