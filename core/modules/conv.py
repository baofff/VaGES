
__all__ = ["Conv2dHWInvariant", "MeanPoolConv", "ConvMeanPool", "UpsampleConv", "ConvUpsample", "ConvNet"]


import torch.nn as nn


class Conv2dHWInvariant(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        assert kernel_size % 2 == 1
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)

    def forward(self, inputs):
        return self.conv(inputs)


class MeanPoolConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.conv(self.pool(inputs))


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()
        self.conv = Conv2dHWInvariant(input_dim, output_dim, kernel_size)
        self.pool = nn.AvgPool2d(2)

    def forward(self, inputs):
        return self.pool(self.conv(inputs))


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)

    def forward(self, inputs):
        return self.conv(self.upsample(inputs))


class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = Conv2dHWInvariant(in_channels, out_channels, kernel_size)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, inputs):
        return self.upsample(self.conv(inputs))


class ConvNet(nn.Module):
    def __init__(self, in_channels, k, af):
        r""" a convolutional net with 2k + 1 layers
        """
        super().__init__()
        net = [nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), af]
        for _ in range(k - 1):
            net.extend([nn.Conv2d(64, 64, 3, stride=1, padding=1), af])
        net.extend([nn.Conv2d(64, 128, 3, stride=2, padding=1), af])
        for _ in range(k - 1):
            net.extend([nn.Conv2d(128, 128, 3, stride=1, padding=1), af])
        net.append(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        return self.net(inputs).flatten(1)
