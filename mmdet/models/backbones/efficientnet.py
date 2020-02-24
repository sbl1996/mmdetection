import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from pytorchcv.model_provider import get_model as ptcv_get_model, calc_tf_padding

from ..registry import BACKBONES


@BACKBONES.register_module
class EfficientNet(nn.Module):
    r"""EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

    version Top1    Top5    Params      FLOPs/2
    b0      22.92   6.75    5,288,548	414.31M     [96, 144, 240, 672, 1280]
    b1      20.73	5.69    7,794,184	732.54M     [96, 144, 240, 672, 1280]
    b2      19.85   5.03    9,109,994	1,051.98M   [96, 144, 288, 720, 1408]
    b3      18.26   4.42    12,233,232	1,928.55M   [144, 192, 288, 816, 1536]
    b4      16.82	3.69	19,341,616	4,607.46M   [144, 192, 336, 960, 1792]
    b5      15.91	3.10	30,389,784	10,695.20M  [144, 240, 384, 1056, 2048]
    b6      15.47	2.96	43,040,704	19,796.24M  [192, 240, 432, 1200, 2304]
    b7      15.13	2.88	66,347,960	39,010.98M  [192, 240, 432, 1200, 2304]
    b8      14.85	2.76	87,413,142	64,541.66M  [192, 336, 528, 1488, 2816]

    Parameters
    ----------
    version : str
        b0, b1, b2, b3, b4, b5, b6, b7, b8 are avaliable.
        Default: b0
    feature_levels (sequence of int): features of which layers to output
        Default: (3, 4, 5)
    """

    def __init__(self,
                 version='b0',
                 feature_levels=(3, 4, 5),
                 pretrained=True,
                 frozen_stages=-1,
                 **kwargs):
        super().__init__()
        self.feature_levels = feature_levels
        self.frozen_stages = frozen_stages
        name = 'efficientnet_%sc' % version
        backbone = ptcv_get_model(name, pretrained=pretrained)
        del backbone.output
        features = backbone.features
        self._kernel_sizes = [3]
        self._strides = [2]
        self.layer1 = nn.Sequential(
            features.init_block.conv,
            features.stage1,
            features.stage2.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage2.unit1.kernel_size)
        self._strides.append(features.stage2.unit1.stride)
        self.layer2 = nn.Sequential(
            features.stage2.unit1.conv2,
            features.stage2.unit1.se,
            features.stage2.unit1.conv3,
            features.stage2[1:],
            features.stage3.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage3.unit1.kernel_size)
        self._strides.append(features.stage3.unit1.stride)
        self.layer3 = nn.Sequential(
            features.stage3.unit1.conv2,
            features.stage3.unit1.se,
            features.stage3.unit1.conv3,
            features.stage3[1:],
            features.stage4.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage4.unit1.kernel_size)
        self._strides.append(features.stage4.unit1.stride)
        self.layer4 = nn.Sequential(
            features.stage4.unit1.conv2,
            features.stage4.unit1.se,
            features.stage4.unit1.conv3,
            features.stage4[1:],
            features.stage5.unit1.conv1,
        )
        self._kernel_sizes.append(features.stage5.unit1.kernel_size)
        self._strides.append(features.stage5.unit1.stride)
        self.layer5 = nn.Sequential(
            features.stage5.unit1.conv2,
            features.stage5.unit1.se,
            features.stage5.unit1.conv3,
            features.stage5[1:],
            features.final_block
        )

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        outs = []
        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[0], self._strides[0]))
        x = self.layer1(x)
        if 1 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[1], self._strides[1]))
        x = self.layer2(x)
        if 2 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[2], self._strides[2]))
        x = self.layer3(x)
        if 3 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[3], self._strides[3]))
        x = self.layer4(x)
        if 4 in self.feature_levels:
            outs.append(x)

        x = F.pad(x, calc_tf_padding(x, self._kernel_sizes[4], self._strides[4]))
        if 5 in self.forward_levels:
            x = self.layer5(x)
            if 5 in self.feature_levels:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
