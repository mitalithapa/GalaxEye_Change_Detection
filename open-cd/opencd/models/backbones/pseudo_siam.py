import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class PseudoSiamChangeFormer(nn.Module):
    def __init__(self, backbone_cfg, in_channels_eo=3, in_channels_sar=1):
        super().__init__()
        cfg_eo = backbone_cfg.copy()
        cfg_eo['in_channels'] = in_channels_eo
        self.encoder_eo = MODELS.build(cfg_eo)

        cfg_sar = backbone_cfg.copy()
        cfg_sar['in_channels'] = in_channels_sar
        self.encoder_sar = MODELS.build(cfg_sar)

        if hasattr(self.encoder_eo, 'layers'):
            for i in range(len(self.encoder_eo.layers)):
                self.encoder_sar.layers[i][1] = self.encoder_eo.layers[i][1]
                self.encoder_sar.layers[i][2] = self.encoder_eo.layers[i][2]
                if i > 0:
                    self.encoder_sar.layers[i][0] = self.encoder_eo.layers[i][0]
        else:
            raise NotImplementedError("This wrapper targets mmseg.MixVisionTransformer.")

    def forward(self, x):
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]
        out_eo = self.encoder_eo(eo)
        out_sar = self.encoder_sar(sar)
        return tuple([torch.cat([e, s], dim=1) for e, s in zip(out_eo, out_sar)])
