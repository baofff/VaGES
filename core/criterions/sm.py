
__all__ = ["ssm", "dsm", "mdsm", "SSM", "DSM", "MDSM", "make_ssm_noise", "make_mdsm_sigmas"]


import torch
import torch.autograd as autograd
import numpy as np
from .base import NaiveCriterion
import core.utils.managers as managers
import core.func as func
from core.lvm.base import LVM


def make_ssm_noise(*size, noise_type, device):
    assert noise_type in ['radermacher', 'gaussian']
    u = torch.randn(*size, device=device)
    if noise_type == 'radermacher':
        u = u.sign()
    return u


def ssm(v, lvm: LVM, noise_type='radermacher', u=None):
    r""" Sliced score matching
    Args:
        v: a batch of data
        lvm: an instance of LVM
        noise_type: the type of the noise
        u: a batch of noise given manually
    """
    if u is None:
        u = make_ssm_noise(*v.shape, noise_type=noise_type, device=v.device)

    with func.RequiresGradContext(v, requires_grad=True):
        log_p = -lvm.free_energy_net(v)
        score = autograd.grad(log_p.sum(), v, create_graph=True)[0]
        loss1 = 0.5 * func.sos(score)

        hvp = autograd.grad((score * u).sum(), v, create_graph=True)[0]
        loss2 = func.inner_product(hvp, u)

    return loss1 + loss2


def dsm(v, lvm: LVM, noise_std, eps=None):
    r""" Denoising score matching
    Args:
        v: a batch of data
        lvm: an instance of LVM
        noise_std: the std of noise
        eps: a batch of standard Gauss noise
    """
    if eps is None:
        eps = torch.randn_like(v, device=v.device)
    v_noised = v + noise_std * eps

    with func.RequiresGradContext(v_noised, requires_grad=True):
        log_p = -lvm.free_energy_net(v_noised)
        score = autograd.grad(log_p.sum(), v_noised, create_graph=True)[0]

    return 0.5 * func.sos(score + eps / noise_std)


def make_mdsm_sigmas(batch_size, sigma_begin, sigma_end, dist, device=None):
    if dist == "linear":
        used_sigmas = torch.linspace(sigma_begin, sigma_end, batch_size, device=device)
    elif dist == "geometrical":
        used_sigmas = torch.logspace(np.log10(sigma_begin), np.log10(sigma_end), batch_size, device=device)
    else:
        raise NotImplementedError
    return used_sigmas


def mdsm(v, lvm: LVM, sigma0, sigma_begin, sigma_end, dist: str, eps=None):
    r""" Multi-level denoising score matching
    Args:
        v: a batch of data
        lvm: an instance of LVM
        sigma0: the base noise std
        sigma_begin: the begin of the range of the noise std
        sigma_end: the end of the range of the noise std
        dist: how the noise std distributed in [sigma_begin, sigma_end]
        eps: a batch of standard Gauss noise
    """
    sigmas = make_mdsm_sigmas(v.size(0), sigma_begin, sigma_end, dist, device=v.device)
    sigmas4v = sigmas.view(len(v), *([1] * (v.dim() - 1)))
    if eps is None:
        eps = torch.randn_like(v).to(v.device)
    v_noised = v + sigmas4v * eps

    with func.RequiresGradContext(v_noised, requires_grad=True):
        log_p = -lvm.free_energy_net(v_noised)
        score = autograd.grad(log_p.sum(), v_noised, create_graph=True)[0]

    return 0.5 * func.sos(score / sigmas4v + eps / sigma0 ** 2)


class SSM(NaiveCriterion):
    def __init__(self,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 noise_type='radermacher'
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm = models.lvm
        self.noise_type = noise_type

    def objective(self, v, **kwargs):
        return ssm(v, self.lvm, self.noise_type)


class DSM(NaiveCriterion):
    def __init__(self,
                 noise_std,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm = models.lvm
        self.noise_std = noise_std

    def objective(self, v, **kwargs):
        return dsm(v, self.lvm, self.noise_std)


class MDSM(NaiveCriterion):
    def __init__(self,
                 sigma0, sigma_begin, sigma_end, dist,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm = models.lvm
        self.sigma0 = sigma0
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        self.dist = dist

    def objective(self, v, **kwargs):
        return mdsm(v, self.lvm, self.sigma0, self.sigma_begin, self.sigma_end, self.dist)
