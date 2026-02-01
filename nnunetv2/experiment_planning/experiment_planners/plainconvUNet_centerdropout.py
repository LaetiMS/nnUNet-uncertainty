from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

import torch.nn as nn

class PlainConvUNet_Center3Dropout(PlainConvUNet):
    """
    PlainConvUNet with Bayesian central-three encoder-decoder dropout.
    Encoder/Decoder implementations are NOT modified.

    Only central stages (default depth=3) have dropout p>0.
    """

    def __init__(self, *args, dropout_op_kwargs=None, bayesian_dropout_cfg=None, **kwargs):
        self.bayesian_cfg = bayesian_dropout_cfg or {}
        self.central_dropout_p = float(self.bayesian_cfg.get("central_p", 0.0))
        self.central_dropout_depth = int(self.bayesian_cfg.get("depth", 3))

        # PURE dropout kwargs — safe for nn.Dropout
        dropout_op_kwargs = dict(dropout_op_kwargs or {})

        super().__init__(
            *args,
            dropout_op_kwargs=dropout_op_kwargs,
            **kwargs
        )

        if self.central_dropout_p > 0:
            self._inject_central_dropout(self.central_dropout_p) #todo: if central_dropout_p is self -> i can do _inject_central_dropout parameterless

        # #for debugging
        # found = 0
        # for m in self.modules():
        #     if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        #         found += 1
        #         print("Dropout p =", m.p)
        #
        # print("Total dropout modules found:", found)
        # print("MODEL ID:", id(self))


    def _inject_central_dropout(self, p: float):

        """
        Injects Bayesian / MC dropout into the central stages of the U-Net.

        Dropout is applied to a band of layers centered around the deepest
        encoder stage (the "bottleneck") and optionally mirrored in the decoder.

        Args:
            p (float): Dropout probability to set in the central stages. Non-central stages
                       will have their dropout probability set to 0.0.

        Behavior:
            - Identifies a central band of encoder stages around the deepest stage,
              width defined by self.bayesian_cfg['depth'] (default 3).
            - Updates all nn.Dropout, nn.Dropout2d, or nn.Dropout3d modules in these stages.
            - Mirrors the dropout in the decoder stages corresponding to the central encoder stages.
            - Leaves all other stages with p=0.0 (no dropout).

        Notes:
            - Only modifies existing dropout layers; does not insert new dropout modules.
            - Recommended central depth for 3D MRI segmentation: 3–4 layers around bottleneck.
            - Useful for MC dropout / Bayesian U-Net uncertainty estimation.
        """
        
        n_enc = len(self.encoder.stages)
        deepest_idx = n_enc - 1
        half_band = self.central_dropout_depth // 2
        central_idxs = set(range(max(deepest_idx - half_band, 0),
                                 min(deepest_idx + half_band + 1, n_enc)))

        # ---------- Encoder ----------
        for i, stage in enumerate(self.encoder.stages):
            stage_p = p if i in central_idxs else 0.0
            for m in stage.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.p = stage_p

        # ---------- Decoder ----------
        for i, stage in enumerate(self.decoder.stages):
            enc_idx = n_enc - i - 2  # mirrored encoder stage
            stage_p = p if enc_idx in central_idxs else 0.0
            for m in stage.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.p = stage_p


    def enable_mc_dropout(self):
        """
        Keeps dropout active during inference while BN stays frozen.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()