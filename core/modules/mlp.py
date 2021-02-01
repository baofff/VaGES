
__all__ = ["TrapezoidMLP", "MLP"]


import torch.nn as nn
import numpy as np
from typing import List


class MLP(nn.Module):
    def __init__(self, n_features_lst: List[int], af):
        super().__init__()
        modules = []
        for i in range(len(n_features_lst) - 1):
            modules.append(nn.Linear(n_features_lst[i], n_features_lst[i + 1]))
            if i < len(n_features_lst) - 2:
                modules.append(af)
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)


class TrapezoidMLP(nn.Module):
    def __init__(self, in_features, out_features, n_layers, af):
        r""" A MLP with n_layers (not including the input layer)
        """
        super(TrapezoidMLP, self).__init__()
        self.af = af
        widths = np.linspace(in_features, out_features, n_layers + 1).astype(np.int)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i != n_layers - 1:
                layers.append(self.af)
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)
