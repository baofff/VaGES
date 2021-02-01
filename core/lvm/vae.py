
__all__ = ["VAE", "GaussVAE", "BernoulliVAE"]


import torch
from .base import LVM
import core.func as func
import math
from core.utils import global_device


class VAE(LVM):
    def __init__(self, h_dim):
        r""" Gauss variational auto-encoder

        p(h) ~ standard Gauss, p(v|h) ~ an amortized distribution
        """
        super(VAE, self).__init__(h_dim=h_dim)
        self.device = global_device()

    def sample_h(self, n_samples):
        return torch.randn(n_samples, self.h_dim).to(self.device)

    def log_joint(self, v, h):
        return func.log_normal(h, 0., 0., 1) + self.log_cpv(v, h)

    def energy_net(self, v, h):
        return self.log_joint(v, h)


class GaussVAE(VAE):
    def __init__(self, param_net, h_dim, std):
        r""" Gauss variational auto-encoder

        p(h) ~ standard Gauss, p(v|h) ~ an amortized Gauss

        Args:
            param_net: returning the mean of a Gauss
        """
        super(GaussVAE, self).__init__(h_dim)
        self.param_net = param_net
        self.std = std
        self.log_std = math.log(self.std)

    def cexpect_v(self, h):
        mean = self.param_net(h.view(-1, self.h_dim))
        return mean.view(*h.shape[:-1], *mean.shape[1:])

    def log_cpv(self, v, h):
        mean = self.cexpect_v(h)
        return func.log_normal(v, mean, math.log(self.std), v.dim() - 1)


class BernoulliVAE(VAE):
    def __init__(self, param_net, h_dim):
        r""" Bernoulli variational auto-encoder

        p(h) ~ standard Gauss, p(v|h) ~ an amortized Bernoulli

        Args:
            param_net: returning the logits of a Bernoulli
        """
        super(BernoulliVAE, self).__init__(h_dim)
        self.param_net = param_net

    def _logits(self, h):
        logits = self.param_net(h.view(-1, self.h_dim))
        return logits.view(*h.shape[:-1], *logits.shape[1:])

    def cexpect_v(self, h):
        return self._logits(h).sigmoid()

    def log_cpv(self, v, h):
        logits = self._logits(h)
        return func.log_bernoulli(v, logits, v.dim() - 1)
