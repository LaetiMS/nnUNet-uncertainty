import numpy as np

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform


# for TTA we want to save the metadata of the spacial transforms so we can apply the inverse and align the predictions for evaluation.



class TTAPipeline:
    def __init__(self, transform: BasicTransform):
        self.transform = transform

    def __call__(self, data):
        """
        Apply TTA transform.
        Returns:
            - augmented data
            - metadata for inversion (flip axes, rotations)
        """
        # For nnU-Net-style ComposeTransforms, you may need to modify transforms to return metadata
        data_aug, metadata = self.transform.apply_with_metadata(data)
        return data_aug, metadata

    def inverse(self, pred, metadata):
        """
        Undo spatial transforms (flip/rotation) so pred aligns with original image.
        """
        if metadata is None:
            return pred
        pred_orig = self.transform.inverse_with_metadata(pred, metadata)
        return pred_orig



class MirrorTransformTTA(MirrorTransform):
    def __init__(self, allowed_axes):
        self.allowed_axes = allowed_axes

    def __call__(self, data):
        # Decide flips
        axes_to_flip = [ax for ax in self.allowed_axes if random.random() < 0.5]
        # Apply flips
        data_flipped = np.flip(data, axes=axes_to_flip).copy()
        # Return both data and metadata
        metadata = {'mirror_axes': axes_to_flip}
        return data_flipped, metadata

    def inverse(self, data, metadata):
        axes_to_flip = metadata.get('mirror_axes', [])
        if axes_to_flip:
            return np.flip(data, axes=axes_to_flip).copy()
        return data

class SpatialTransform(SpatialTransform):
    def __init__(self, allowed_axes):
        self.allowed_axes = allowed_axes

    def __call__(self, data):
        # Decide flips
        axes_to_flip = [ax for ax in self.allowed_axes if random.random() < 0.5]
        # Apply flips
        data_flipped = np.flip(data, axes=axes_to_flip).copy()
        # Return both data and metadata
        metadata = {'mirror_axes': axes_to_flip}
        return data_flipped, metadata

    def inverse(self, data, metadata):
        axes_to_flip = metadata.get('mirror_axes', [])
        if axes_to_flip:
            return np.flip(data, axes=axes_to_flip).copy()
        return data



class SpatialTransformTTA(SpatialTransform):
    def __call__(self, data):
        # Generate random rotation/scale
        rot_x = np.random.uniform(*self.rotation_range_x)
        rot_y = np.random.uniform(*self.rotation_range_y)
        scale = np.random.uniform(*self.scale_range)

        # Apply transform
        data_transformed = apply_affine_transform(data, rot_x, rot_y, scale)

        # Save metadata for inversion
        metadata = {
            'rotation_x': rot_x,
            'rotation_y': rot_y,
            'scale': scale
        }
        return data_transformed, metadata

    def inverse(self, data, metadata):
        # Apply inverse affine using metadata
        return apply_inverse_affine(data,
                                    rotation_x=-metadata['rotation_x'],
                                    rotation_y=-metadata['rotation_y'],
                                    scale=1/metadata['scale'])

    @staticmethod
    def get_tta_extreme_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
    ) -> BasicTransform:
        """
        Its goal is not to improve mean segmentation accuracy, but to estimate predictive uncertainty by probing how sensitive the trained model is to plausible variations of the same input image.
        For this we reuse some of the transforms used in training but increase its probability / range.

        The pipeline is designed so that:
        - the expected prediction remains unbiased
        - variance in predictions reflects model ambiguity, not augmentation artifacts
        - all perturbations are input-only and label-preserving
        """

        transforms = []

        # --- Handle 2D-in-3D case (same as training) ---
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # --- Conservative spatial perturbations ---
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.5, #instead of 0.2
                rotation=(-0.15, 0.15),     # ~±8.5° # instead of rotation_for_DA
                p_scaling=0.5, #instead of 0.2
                scaling=(0.9, 1.1), #instead of (0.7, 1.4)
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # --- Noise (strong driver of uncertainty) ---
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    noise_variance=(0, 0.2), # instead of (0, 0.1)
                    p_per_channel=1,
                    synchronize_channels=True
                ),
                apply_probability=0.4 # instead of 0.1
            )
        )

        # --- Blur ---
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 2.0), # instead of (0.5, 1.)
                    synchronize_channels=True, # instead of False
                    synchronize_axes=True, # instead of False
                    p_per_channel=1  #benchmark False instead of True
                ),
                apply_probability=0.3 # instead of 0.15
            )
        )
        # --- Brightness ---
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=(0.7, 1.3), # instead of BGContrast((0.75, 1.25)) -> we remove BGContrast to avoid biasing to background / foreground,
                    synchronize_channels=True, # instead of False
                    p_per_channel=1
                ),
                apply_probability=0.4 # instead of 0.15
            )
        )

        # --- Contrast ---
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=(0.6, 1.4), # instead of BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ),
                apply_probability=0.4 # instead of 0.15
            )
        )

        # --- Low resolution ---
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.4, 1.0),
                    synchronize_channels=True,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    p_per_channel=1
                ),
                apply_probability=0.4
            )
        )

        # --- Gamma ---
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=(0.5, 1.8), # BGContrast((0.75, 1.25)),
                    p_invert_image=0.5, # instead of 1 or 0
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=0 # instead of 1
                ),
                apply_probability=0.4 # instead of 1 or 0.3
            )
        )

        #  is_cascaded is not implemented btw:)


        return ComposeTransforms(transforms)



