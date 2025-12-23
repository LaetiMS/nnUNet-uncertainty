# have a look at https://github.com/smriti-joshi/muvi/blob/main/scripts/infer_with_test_time_adaptation.py#L77
# keep in mind that her method focused on 2d and is only compatible with single fold evaluation -> so using fold_0 :)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# for MCDropout predictor should enable_dropout_layers
# for Layered ensembles predictor should enable enable_deep_supervision = True during prediction
#             logits = network(x)
#             # logits is now a list: [deepest, ..., final]
#             layer_logits = logits[-3:] # if i only keep the last few layers
# for TTA

class nnUNetPredictor_TTA_extended(nnUNetPredictor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
