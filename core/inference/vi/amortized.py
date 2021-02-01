
__all__ = ["AmortizedGauss", "AmortizedBernoulli"]


import torch
import torch.nn as nn
import core.func as func


class AmortizedGauss(nn.Module):
    def __init__(self, param_net, h_dim):
        r"""

        Args:
            param_net: returning the mean and log_std of a Gauss
        """
        super().__init__()
        self.param_net = param_net
        self.h_dim = h_dim

    def forward(self, v):
        r""" higher package only support modules with forward
        """
        return self.param_net(v)

    def expect(self, v):
        r""" E[h|v]

        Args:
            v: batch_size * v_shape
        """
        return self.forward(v)[0]

    def log_q(self, h, v):
        r""" log q(h|v)

        Args:
            h: (n_particles *) batch_size * h_dim
            v: batch_size * v_shape
        """
        mean, log_std = self.forward(v)
        return func.log_normal(h, mean, log_std, 1)

    def implicit_net(self, v, n_particles=None, eps=None):
        r""" Sample from q(h|v) as an implicit model
        """
        assert not (n_particles is None and eps is None)
        if eps is None:
            eps = torch.randn(n_particles, len(v), self.h_dim, device=v.device)
        mean, log_std = self.forward(v)
        return mean + log_std.exp() * eps

    def symmetric_sample(self, v, n_particles=None, eps=None):
        if eps is None:
            assert n_particles is not None and n_particles % 2 == 0
            eps = torch.randn(n_particles // 2, len(v), self.h_dim).to(v.device)
        eps = torch.cat([eps, -eps], dim=0)
        return self.implicit_net(v, eps=eps)

    def implicit_net_log_q(self, v, n_particles=None, eps=None):
        assert not (n_particles is None and eps is None)
        mean, log_std = self.forward(v)
        if eps is None:
            eps = torch.randn(n_particles, len(v), self.h_dim).to(v.device)
        h = mean + log_std.exp() * eps
        log_q = func.log_normal(h, mean, log_std, 1)
        return h, log_q


def _sample_gumbel(*size, infinitesimal=1e-20, device=None):
    r""" Sample from the standard gumbel distribution
    """
    return -torch.log(-torch.log(torch.rand(*size, device=device) + infinitesimal) + infinitesimal)


def _gumbel_softmax(logits, eps, temperature):
    r"""
    Args:
        eps: samples from the standard gumbel distribution
    """
    return ((logits + eps) / temperature).softmax(dim=-1)


def _bernoulli_reparameterization(eps, logits, temperature):
    logits = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
    log_pi = logits.log_softmax(dim=-1)
    return _gumbel_softmax(log_pi, eps, temperature).select(-1, 0)


class AmortizedBernoulli(nn.Module):
    def __init__(self, param_net, h_dim, temperature=0.1):
        r"""

        Args:
            param_net: returning the logits of a Bernoulli
        """
        super().__init__()
        self.param_net = param_net
        self.h_dim = h_dim
        self.temperature = temperature

    def forward(self, v):
        r""" higher package only support modules with forward
        """
        return self.param_net(v)

    def expect(self, v):
        return self.forward(v).sigmoid()

    def log_q(self, h, v):
        r""" log q(h|v)

        Args:
            h: (n_particles *) batch_size * h_dim
            v: batch_size * v_shape
        """
        logits = self.forward(v)
        return func.log_bernoulli(h, logits, 1)

    def implicit_net(self, v, n_particles=None, eps=None):
        r""" Sample from q(h|v) as an implicit model
        """
        assert not (n_particles is None and eps is None)
        if eps is None:
            eps = _sample_gumbel(n_particles, len(v), self.h_dim, 2, device=v.device)
        logits = self.forward(v)
        return _bernoulli_reparameterization(eps, logits, self.temperature)

    def implicit_net_log_q(self, v, n_particles=None, eps=None):
        assert not (n_particles is None and eps is None)
        if eps is None:
            eps = _sample_gumbel(n_particles, len(v), self.h_dim, 2, device=v.device)
        logits = self.forward(v)
        h = _bernoulli_reparameterization(eps, logits, self.temperature)
        log_q = func.log_bernoulli(h, logits, 1)
        return h, log_q
