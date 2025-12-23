from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

import torch.nn as nn

def central_indices(n: int):
    c = n // 2
    return {c - 1, c, c + 1}


class PlainConvUNet_Center3Dropout(PlainConvUNet):
    """
    PlainConvUNet with Bayesian central-three encoder-decoder dropout.
    Encoder/Decoder implementations are NOT modified.
    """

    def __init__(self, *args, dropout_op_kwargs=None, **kwargs):
        central_dropout_p = 0.0
        if dropout_op_kwargs is not None and 'central_dropout_p' in dropout_op_kwargs:
            central_dropout_p = dropout_op_kwargs.pop('central_dropout_p')
        self.central_dropout_p = central_dropout_p
        # Call parent init with the modified kwargs
        super().__init__(*args, dropout_op_kwargs=dropout_op_kwargs, **kwargs)

        if central_dropout_p > 0:
            self._inject_central_dropout(self.central_dropout_p) #todo: if central_dropout_p is self -> i can do _inject_central_dropout parameterless

    def _inject_central_dropout(self, p: float):
        n_enc = len(self.encoder.stages)
        enc_centers = central_indices(n_enc)

        # ---------- Encoder ----------
        for i, stage in enumerate(self.encoder.stages):
            for m in stage.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    if i in enc_centers:
                        m.p = p
                    else:
                        m.p = 0.0

        # ---------- Decoder ----------
        for i, stage in enumerate(self.decoder.stages):
            dec_idx = n_enc - i - 2  # mirror depth
            for m in stage.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    if dec_idx in enc_centers:
                        m.p = p
                    else:
                        m.p = 0.0

    def enable_mc_dropout(self):
        """
        Keeps dropout active during inference while BN stays frozen.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()