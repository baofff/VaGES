import torch
import torch.autograd as autograd
from core.criterions.base import NaiveCriterion
import core.func as func
import core.utils.managers as managers
import numpy as np


def rbf_kernel(x, y, bandwidth=0.1):
    assert x.shape[0] == y.shape[0]
    dis = func.sos(x - y, start_dim=1)
    kernel = torch.exp(-dis / (2 * bandwidth ** 2)).unsqueeze(dim=1)
    gradient = kernel * (x - y) * (-1 / bandwidth ** 2)
    return kernel, gradient


class KSD(NaiveCriterion):
    def __init__(self,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm = models.lvm

    def objective(self, v, **kwargs):
        bs = v.size(0)
        assert bs % 2 == 0
        v, vp = v[bs // 2:], v[: bs // 2]

        with func.RequiresGradContext(v, requires_grad=True):
            log_p = -self.lvm.free_energy_net(v)
            score_v = autograd.grad(log_p.sum(), v, retain_graph=True)[0].detach()

        with func.RequiresGradContext(vp, requires_grad=True):
            log_p = -self.lvm.free_energy_net(vp)
            score_vp = autograd.grad(log_p.sum(), vp, create_graph=True)[0]

        ker, grad_ker = rbf_kernel(v, vp)
        z = (ker * score_v + grad_ker).detach()
        return func.inner_product(z, score_vp)


def free_energy_net_is(v, lvm, n_particles):
    h = torch.bernoulli(torch.zeros(n_particles, v.shape[0], lvm.h_dim) + 0.5).to(v.device)
    e = lvm.energy_net(v, h)
    return -func.logsumexp(-e, dim=0) + lvm.h_dim * np.log(0.5) + np.log(n_particles)


class ISKSD(NaiveCriterion):
    def __init__(self,
                 n_particles: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.n_particles = n_particles
        self.lvm = models.lvm

    def objective(self, v, **kwargs):
        bs = v.size(0)
        assert bs % 2 == 0
        v, vp = v[bs // 2:], v[: bs // 2]

        with func.RequiresGradContext(v, requires_grad=True):
            log_p = -free_energy_net_is(v, self.lvm, self.n_particles)
            score_v = autograd.grad(log_p.sum(), v, retain_graph=True)[0].detach()

        with func.RequiresGradContext(vp, requires_grad=True):
            log_p = -free_energy_net_is(vp, self.lvm, self.n_particles)
            score_vp = autograd.grad(log_p.sum(), vp, create_graph=True)[0]

        ker, grad_ker = rbf_kernel(v, vp)
        z = (ker * score_v + grad_ker).detach()
        return func.inner_product(z, score_vp)
