from core.lvm import GRBM
import core.func as func
import torch
import numpy as np


class InitGRBM(GRBM):
    def __init__(self, target_grbm: GRBM):
        super().__init__(target_grbm.v_dim, target_grbm.h_dim, False)
        del self.log_std, self.b_v, self.b_h, self.W
        self.log_std = target_grbm.log_std.data.clone()
        self.b_v = target_grbm.b_v.data.clone()
        self.b_h = torch.zeros(self.h_dim, device=self.device)
        self.W = torch.zeros(self.v_dim, self.h_dim, device=self.device)

        # the log partition w.r.t. the free energy net
        self.log_partition = self.log_std.sum() + 0.5 * np.log(2 * np.pi) * self.v_dim + np.log(2.) * self.h_dim

    def sample(self, n_samples):
        eps = torch.randn(n_samples, self.v_dim, device=self.device)
        return self.b_v + self.log_std.exp() * eps


class TransitGRBM(GRBM):
    def __init__(self, target_grbm: GRBM):
        super().__init__(target_grbm.v_dim, target_grbm.h_dim, False)
        del self.log_std, self.b_v, self.b_h, self.W
        self.log_std = target_grbm.log_std.data.clone()
        self.b_v = target_grbm.b_v.data.clone()
        self.b_h = None
        self.W = None
        self.target_b_h = target_grbm.b_h.data.clone()
        self.target_W = target_grbm.W.data.clone()

    def set_status(self, weight):
        self.b_h = weight * self.target_b_h
        self.W = weight * self.target_W

    def vhv(self, v):
        return self.csample_v(self.csample_h(v, n_particles=1).squeeze(dim=0))


class AIS4GRBM(object):
    def __init__(self, grbm: GRBM, n_samples=2000, n_transit=2000):
        self.grbm = grbm
        self.n_samples = n_samples
        self.n_transit = n_transit
        self.log_partition = None

    def update_log_partition(self):
        init_grbm = InitGRBM(self.grbm)
        weights = torch.linspace(0., 1., self.n_transit)
        v = init_grbm.sample(self.n_samples)
        transit_grbm = TransitGRBM(self.grbm)
        log_w = init_grbm.free_energy_net(v)
        for weight in weights[:-1]:
            transit_grbm.set_status(weight)
            log_w -= transit_grbm.free_energy_net(v)
            v = transit_grbm.vhv(v)
            log_w += transit_grbm.free_energy_net(v)
        transit_grbm.set_status(weights[-1])
        log_w -= transit_grbm.free_energy_net(v)
        self.log_partition = func.logsumexp(log_w, dim=0) - np.log(self.n_samples) + init_grbm.log_partition

    def estimate_ll(self, v):
        assert self.log_partition is not None
        return -self.grbm.free_energy_net(v) - self.log_partition
