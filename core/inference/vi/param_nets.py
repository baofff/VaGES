
__all__ = ["ConvParamNet", "MLPResidualParamNet"]


import torch.nn as nn
import core.modules as modules
import core.utils.diagnose as diagnose
from typing import List


class ConvParamNet(nn.Module):
    def __init__(self, v_shape: List[int], h_dim: int, k: int, af=nn.LeakyReLU(0.2)):
        super().__init__()
        self.af = af
        self.main = modules.ConvNet(in_channels=v_shape[0], k=k, af=af)
        n_features = diagnose.probe_output_shape(self.main, v_shape)
        assert len(n_features) == 1
        self.mean = nn.Linear(n_features[0], h_dim)
        self.log_std = nn.Linear(n_features[0], h_dim)

    def forward(self, v):
        m = self.af(self.main(v))
        mean = self.mean(m)
        log_std = self.log_std(m)
        return mean, log_std


class MLPResidualParamNet(nn.Module):
    def __init__(self, v_dim: int, h_dim: int, n_hiddens_lst: List[int], af=nn.LeakyReLU(0.2)):
        super().__init__()
        self.af = af
        self.main = modules.MLPResidualNet([v_dim] + n_hiddens_lst, af)
        n_features = n_hiddens_lst[-1]
        self.mean = nn.Linear(n_features, h_dim)
        self.log_std = nn.Linear(n_features, h_dim)

    def forward(self, v):
        m = self.af(self.main(v))
        mean = self.mean(m)
        log_std = self.log_std(m)
        return mean, log_std
