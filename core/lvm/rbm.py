
__all__ = ["GRBM"]


import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import core.func as func
from .base import LVM
from core.utils import global_device


class GRBM(LVM):
    def __init__(self, v_dim, h_dim, fix_std, std=None):
        r""" Gauss RBM
        """
        super().__init__(v_dim=v_dim, h_dim=h_dim)
        self.device = global_device()
        if fix_std:
            assert std is not None
            self.log_std = torch.ones(v_dim, device=self.device) * np.log(std)
        else:
            self.log_std = nn.Parameter(torch.zeros(v_dim), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros(h_dim), requires_grad=True)
        self.b_v = nn.Parameter(torch.zeros(v_dim), requires_grad=True)
        self.W = nn.Parameter(torch.zeros(v_dim, h_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_h, -bound, bound)
        nn.init.uniform_(self.b_v, -bound, bound)

    def cexpect_h(self, v):
        return (self.b_h + v @ self.W).sigmoid()

    def cexpect_v(self, h):
        return self.b_v + (h @ self.W.t()) * (self.log_std * 2).exp()

    def csample_h(self, v, n_particles):
        return func.duplicate(self.cexpect_h(v), n_particles).bernoulli()

    def csample_v(self, h):
        eps = torch.randn(*h.shape[:-1], self.v_dim, device=h.device)
        mean = self.cexpect_v(h)
        v = mean + self.log_std.exp() * eps
        return v

    def log_cpv(self, v, h):
        mean = self.cexpect_v(h)
        return func.log_normal(v, mean, self.log_std, n_data_dim=1)

    def log_cph(self, h, v):
        logits = self.b_h + v @ self.W
        return func.log_bernoulli(h, logits, n_data_dim=1)

    def energy_net(self, v, h):
        v_part = 0.5 * func.sos((v - self.b_v) * (-self.log_std).exp())
        h_part = - h @ self.b_h
        vh_part = - ((v @ self.W) * h).sum(dim=-1)
        return v_part + h_part + vh_part

    def free_energy_net(self, v):
        a = 0.5 * func.sos((v - self.b_v) * (-self.log_std).exp())
        b = F.softplus(self.b_h + v @ self.W).sum(dim=-1)
        return a - b
