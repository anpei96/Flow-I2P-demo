from typing import Union

import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer
from image_backbone  import BasicBlock

class ImageBackbone_tiny(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        dilation: int = 1,
        norm_cfg: Union[str, dict] = "BatchNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        self.encoder0 = ConvBlock(
            in_channels,
            32,
            kernel_size=3,
            padding=1,
            stride=1,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.encoder1 = nn.Sequential(
            ConvBlock(
                32,
                64,
                kernel_size=3,
                padding=1,
                stride=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.encoder2 = ConvBlock(
            64,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        x = self.encoder0(x)  
        x = self.encoder1(x)  
        x = self.encoder2(x) 
        return x