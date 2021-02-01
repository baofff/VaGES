
__all__ = ["ResidualBlock", "ResidualNetDown", "ResidualNetUp",
           "MLPResidualBlock", "MLPResidualNet"]


import torch.nn as nn
from . import conv
from typing import List


################################################################################
# ResNet based on conv
################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resample, af=nn.ELU()):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resample = resample
        self.af = af

        self.conv1 = conv.Conv2dHWInvariant(in_channels, in_channels, kernel_size=kernel_size, bias=False)
        if resample == 'down':
            self.conv2 = conv.ConvMeanPool(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = conv.MeanPoolConv(in_channels, out_channels, kernel_size=1)
        elif resample == "up":
            self.conv2 = conv.ConvUpsample(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = conv.UpsampleConv(in_channels, out_channels, kernel_size=1)
        elif resample == 'none':
            self.conv2 = conv.Conv2dHWInvariant(in_channels, out_channels, kernel_size=kernel_size)
            self.conv_shortcut = conv.Conv2dHWInvariant(in_channels, out_channels, kernel_size=1)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = inputs
        outputs = self.af(outputs)
        outputs = self.conv1(outputs)
        outputs = self.af(outputs)
        outputs = self.conv2(outputs)
        return outputs + self.conv_shortcut(inputs)


class ResidualNetDown(nn.Module):
    def __init__(self, in_channels: int, channels: int, k: int, af=nn.ELU()):
        r""" Conv block -> flattened feature
        Args:
            k: there will be 3k residual blocks, or 6k conv blocks
        """
        super().__init__()
        modules = [conv.Conv2dHWInvariant(in_channels, channels, 3)]
        for i in [1, 2, 4]:
            modules.append(ResidualBlock(i * channels, 2 * i * channels, 3, 'down', af))
            for _ in range(k - 1):
                modules.append(ResidualBlock(2 * i * channels, 2 * i * channels, 3, 'none', af))
        self.down = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.down(inputs).flatten(1)


class ResidualNetUp(nn.Module):
    def __init__(self, in_features: int, channels: int, out_channels: int, out_width: int, k: int, af=nn.ELU()):
        r""" flattened feature -> Conv block
        Args:
            k: there will be 3k residual blocks, or 6k conv blocks
        """
        assert out_width % 8 == 0
        super().__init__()
        self.out_width = out_width
        self.channels = channels
        self.linear = nn.Linear(in_features, (out_width // 8) * (out_width // 8) * 8 * channels)
        modules = []
        for i in [4, 2, 1]:
            for _ in range(k - 1):
                modules.append(ResidualBlock(2 * i * channels, 2 * i * channels, 3, 'none', af))
            modules.append(ResidualBlock(2 * i * channels, i * channels, 3, 'up', af))
        modules.append(af)
        modules.append(conv.Conv2dHWInvariant(channels, out_channels, 3))
        self.up = nn.Sequential(*modules)

    def forward(self, inputs):
        outputs = self.linear(inputs).view(inputs.size(0), 8 * self.channels, self.out_hw // 8, self.out_hw // 8)
        return self.up(outputs)


################################################################################
# ResNet based on mlp
################################################################################

class MLPResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_units: int, af=nn.ELU()):
        super().__init__()
        self.af = af
        self.linear1 = nn.Linear(in_features, hidden_units)
        self.linear2 = nn.Linear(hidden_units, out_features)
        self.short_cut = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        outputs = self.af(self.linear1(inputs))
        outputs = self.af(self.linear2(outputs))
        return outputs + self.short_cut(inputs)


class MLPResidualNet(nn.Module):
    def __init__(self, n_features_lst: List[int], af=nn.ELU()):
        super().__init__()
        modules = []
        for i in range(len(n_features_lst) - 1):
            modules.append(MLPResidualBlock(n_features_lst[i], n_features_lst[i + 1], n_features_lst[i], af))
            if i < len(n_features_lst) - 2:
                modules.append(af)
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)
