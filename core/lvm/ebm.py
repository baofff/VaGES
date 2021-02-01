
__all__ = ["EBM", "EBLVMACP"]


from .base import LVM
import torch.nn as nn
from core.utils import diagnose
import torch
import core.func as func


class EBM(LVM):
    def __init__(self, feature_net, scalar_net, v_shape=None, v_dim=None):
        super().__init__(v_shape=v_shape, v_dim=v_dim)
        self.feature_net = nn.DataParallel(feature_net)
        self.scalar_net = nn.DataParallel(scalar_net)

    def free_energy_net(self, v):
        return self.scalar_net(self.feature_net(v))


class EBLVMACP(LVM):
    def __init__(self, feature_net, scalar_net, v_shape=None, v_dim=None, h_dim=None):
        r""" The energy-based latent variable model with an additive coupling layer
        """
        super().__init__(v_shape=v_shape, v_dim=v_dim, h_dim=h_dim)
        n_features = diagnose.probe_output_shape(feature_net, self.v_shape)
        assert len(n_features) == 1
        self.feature_net = nn.DataParallel(feature_net)
        self.linear = nn.Linear(n_features[0], self.h_dim)
        self.scalar_net = nn.DataParallel(scalar_net)

    def acps(self, fh, h):
        if h.dim() == 2:
            cp = torch.cat([fh + h, h], dim=-1)
            return self.scalar_net(cp)
        elif h.dim() == 3:
            assert fh.size(0) == h.size(1)
            n_particles, batch_size = h.size(0), h.size(1)
            fh = func.duplicate(fh, n_particles).flatten(0, 1)
            h = h.flatten(0, 1)
            cp = torch.cat([fh + h, h], dim=-1)
            return self.scalar_net(cp).view(n_particles, batch_size)
        else:
            raise ValueError

    def energy_net(self, v, h):
        fh = self.linear(self.feature_net(v))
        return self.acps(fh, h)

    def energy_net_cut_fn_v(self, v):
        fh = self.linear(self.feature_net(v))
        return lambda _h: self.acps(fh, _h)
