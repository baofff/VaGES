from tqdm import tqdm
import core.func as func
import torch
import torch.autograd as autograd
from typing import List, Union
import numpy as np
import core.utils as utils
import functools
import math


class SGLDInfo(object):
    def __init__(self, samples, init=None, middle_states=None, fn_vals=None):
        self.samples = samples
        self.init = init
        self.middle_states = middle_states
        self.fn_vals = fn_vals


################################################################################
# sgld
################################################################################

def sgld(fn, init: list, alpha: float, sigma: float, n_steps: int, grad_max_norms: Union[float, List[float]] = None, vis: bool = True):
    r""" x' = x - 0.5 * alpha * ▽x fn(x) + sigma * eps,  eps ~ N(0, I)
    Args:
        fn: the energy function
        init: the initial state
        alpha: the step length
        sigma: the noise level
        n_steps: the number of sgld steps
        grad_max_norms: the maximum gradient norm for gradient clip
        vis: whether to show the sample process
    """
    if not isinstance(grad_max_norms, list):
        grad_max_norms = [grad_max_norms] * len(init)
    assert len(init) == len(grad_max_norms)

    inputs = init
    for _ in tqdm(range(n_steps), desc="sgld", disable=not vis):
        with func.RequiresGradContext(*inputs, requires_grad=True):
            fn_val = fn(*inputs)
            grads = autograd.grad(fn_val.sum(), inputs)

        old_inputs, inputs = inputs, []
        for x, grad, max_norm in zip(old_inputs, grads, grad_max_norms):
            eps = torch.randn_like(x)
            if max_norm is not None:
                utils.clip_grad_element_wise_(grad, max_norm)
            inputs.append(x.detach() - 0.5 * alpha * grad + sigma * eps)
    return inputs


def sgld_decay(fn, init: list, init_alpha: float, n_steps: int, grad_max_norms: Union[float, List[float]] = None, vis: bool = True):
    if not isinstance(grad_max_norms, list):
        grad_max_norms = [grad_max_norms] * len(init)
    assert len(init) == len(grad_max_norms)

    inputs = init
    decay_fn = lambda _i: math.log(_i) / _i
    for i in tqdm(range(n_steps), desc="sgld_decay", disable=not vis):
        alpha = init_alpha * decay_fn(i + 3) / decay_fn(3)
        with func.RequiresGradContext(*inputs, requires_grad=True):
            fn_val = fn(*inputs)
            grads = autograd.grad(fn_val.sum(), inputs)
        old_inputs, inputs = inputs, []
        for x, grad, max_norm in zip(old_inputs, grads, grad_max_norms):
            eps = torch.randn_like(x)
            if max_norm is not None:
                utils.clip_grad_element_wise_(grad, max_norm)
            inputs.append(x.detach() - 0.5 * alpha * grad + alpha ** 0.5 * eps)
    return inputs


def sgld_annealed(fn, init: list, alpha: float, sigmas: List[float], denoise: bool = True, sigma0: float = None,
                  grad_max_norms: Union[float, List[float]] = None, vis: bool = True):
    if not isinstance(grad_max_norms, list):
        grad_max_norms = [grad_max_norms] * len(init)
    assert len(init) == len(grad_max_norms)

    inputs = init
    for i, sigma in tqdm(enumerate(sigmas), desc="sgld_annealed", disable=not vis):
        with func.RequiresGradContext(*inputs, requires_grad=True):
            fn_val = fn(*inputs)
            grads = autograd.grad(fn_val.sum(), inputs)

        old_inputs, inputs = inputs, []
        for x, grad, max_norm in zip(old_inputs, grads, grad_max_norms):
            eps = torch.randn_like(x)
            if max_norm is not None:
                utils.clip_grad_element_wise_(grad, max_norm)
            inputs.append(x.detach() - 0.5 * alpha * grad + sigma * eps)
    if denoise:
        assert sigma0 is not None
        inputs, _ = ss_denoise(fn, inputs, sigma0)
    return inputs


################################################################################
# Annealed sgld
################################################################################

def ss_denoise(fn, noised, sigma0):
    r""" Single step denoise
    Args:
        fn: the energy function
        noised: the noised state
        sigma0: the denoise step
    """
    sigma02 = sigma0 ** 2
    with func.RequiresGradContext(*noised, requires_grad=True):
        fn_val = fn(*noised)
        grads = autograd.grad(fn_val.sum(), noised)
    denoised = [x.detach() - sigma02 * grad for x, grad in zip(noised, grads)]
    return denoised, fn_val


def annealed_sgld(fn, init: list, sigma: float, Ts: List[float], denoise: bool = True, sigma0: float = None,
                  share_random: bool = False, record_middle_states: bool = False, n_middle_states: int = None,
                  vis: bool = True):
    r"""
    Args:
        fn: the energy function
        init: the initial state
        sigma: the noise level
        denoise: whether to denoise the sample
        sigma0: the denoise step
        share_random: whether to share the randomness among a batch
        Ts: the annealing schedule
        record_middle_states: whether to record middle states
        n_middle_states: the number of middle states
        vis: whether to show the sample process
    """
    sigma2 = sigma ** 2
    middle_states, fn_vals = [], []
    inputs = init
    period = (len(Ts) - 1) // n_middle_states if record_middle_states else None
    for i, T in tqdm(enumerate(Ts), desc="annealed sgld", disable=not vis):
        with func.RequiresGradContext(*inputs, requires_grad=True):
            fn_val = fn(*inputs)
            grads = autograd.grad(fn_val.sum(), inputs)

        fn_vals.append(fn_val.detach())
        old_inputs, inputs = inputs, []
        for x, grad in zip(old_inputs, grads):
            eps = func.duplicate(torch.randn(*x.shape[1:], device=x.device), x.size(0)) if share_random else torch.randn_like(x)
            inputs.append(x.detach() - 0.5 * sigma2 * grad + T ** 0.5 * sigma * eps)
        if record_middle_states and (i + 1) % period == 0:
            middle_states.append(inputs)
    if denoise:
        assert sigma0 is not None
        inputs, _ = ss_denoise(fn, inputs, sigma0)
        if record_middle_states:
            middle_states[-1] = inputs
    return SGLDInfo(samples=inputs, init=init, fn_vals=fn_vals,
                    middle_states=None if not record_middle_states else middle_states)


def ss_denoise_impainting(fn, noised, keep, sigma0):
    r""" Single step denoise
    Args:
        fn: the energy function
        noised: the noised state
        keep: whether to keep the pixel
        sigma0: the denoise step
    """
    change = 1. - keep
    sigma02 = sigma0 ** 2
    with func.RequiresGradContext(*noised, requires_grad=True):
        fn_val = fn(*noised)
        grads = autograd.grad(fn_val.sum(), noised)
    denoised = [x.detach() - sigma02 * grad * change for x, grad in zip(noised, grads)]
    return denoised, fn_val


def annealed_sgld_impainting(fn, init: list, keep: torch.Tensor, sigma: float, Ts: List[float], denoise: bool = True, sigma0: float = None,
                             share_random: bool = False, record_middle_states: bool = False, n_middle_states: int = None,
                             vis: bool = True):
    r"""
    Args:
        fn: the energy function
        init: the initial state
        keep: whether to keep the pixel
        sigma: the noise level
        denoise: whether to denoise the sample
        sigma0: the denoise step
        share_random: whether to share the randomness among a batch
        Ts: the annealing schedule
        record_middle_states: whether to record middle states
        n_middle_states: the number of middle states
        vis: whether to show the sample process
    """
    change = 1. - keep
    sigma2 = sigma ** 2
    middle_states, fn_vals = [], []
    inputs = init
    period = (len(Ts) - 1) // n_middle_states if record_middle_states else None
    for i, T in tqdm(enumerate(Ts), desc="annealed sgld impainting", disable=not vis):
        with func.RequiresGradContext(*inputs, requires_grad=True):
            fn_val = fn(*inputs)
            grads = autograd.grad(fn_val.sum(), inputs)

        fn_vals.append(fn_val.detach())
        old_inputs, inputs = inputs, []
        for x, grad in zip(old_inputs, grads):
            eps = func.duplicate(torch.randn(*x.shape[1:], device=x.device),
                                 x.size(0)) if share_random else torch.randn_like(x)
            inputs.append(x.detach() - 0.5 * sigma2 * grad * change + T ** 0.5 * sigma * eps * change)
        if record_middle_states and (i + 1) % period == 0:
            middle_states.append(inputs)
    if denoise:
        assert sigma0 is not None
        inputs, _ = ss_denoise_impainting(fn, inputs, keep, sigma0)
        if record_middle_states:
            middle_states[-1] = inputs
    return SGLDInfo(samples=inputs, init=init, fn_vals=fn_vals,
                    middle_states=None if not record_middle_states else middle_states)


def geometric_annealed_scheme(steps_max, steps_annealed, steps_min, Tmax, Tmin):
    assert Tmax >= Tmin
    Ts_max = Tmax * np.ones((steps_max,))
    Ts_annealed = np.geomspace(Tmax, Tmin, steps_annealed + 1)[:-1]
    Ts_min = Tmin * np.linspace(1, 0, steps_min)
    Ts = list(np.concatenate((Ts_max, Ts_annealed, Ts_min), axis=0))
    return Ts


def geometric_annealed_sgld(fn, init: list, sigma: float,
                            steps_max: int, steps_annealed: int, steps_min: int, Tmax: float, Tmin: float,
                            denoise: bool = True, sigma0: float = None, share_random: bool = False,
                            record_middle_states: bool = False, n_middle_states: int = None, vis: bool = True):
    Ts = geometric_annealed_scheme(steps_max, steps_annealed, steps_min, Tmax, Tmin)
    return annealed_sgld(fn, init, sigma, Ts, denoise, sigma0, share_random, record_middle_states, n_middle_states, vis)


class GeometricAnnealedSGLD(object):
    def __init__(self, lvm, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200, Tmax=100, Tmin=1,
                 denoise=True, sigma0=0.1, record_middle_states=False, n_middle_states=None, vis=True):
        self.lvm = lvm
        self.sigma = sigma
        self.denoise = denoise
        self.sigma0 = sigma0
        self.record_middle_states = record_middle_states
        self.n_middle_states = n_middle_states
        self.vis = vis
        self.Ts = geometric_annealed_scheme(steps_max, steps_annealed, steps_min, Tmax, Tmin)
        self.device = utils.global_device()

    def sample(self, n_samples):
        init_v = 0.5 + torch.randn(n_samples, *self.lvm.v_shape, device=self.device)
        init = [init_v]

        return annealed_sgld(fn=self.lvm.free_energy_net, init=init, sigma=self.sigma, Ts=self.Ts,
                             denoise=self.denoise, sigma0=self.sigma0, share_random=False,
                             record_middle_states=self.record_middle_states, n_middle_states=self.n_middle_states,
                             vis=self.vis)

    def sample_joint(self, n_samples):
        init_v = 0.5 + torch.randn(n_samples, *self.lvm.v_shape, device=self.device)
        init_h = 0.5 + torch.randn(n_samples, self.lvm.h_dim, device=self.device)
        init = [init_v, init_h]

        return annealed_sgld(fn=self.lvm.energy_net, init=init, sigma=self.sigma, Ts=self.Ts,
                             denoise=self.denoise, sigma0=self.sigma0, share_random=False,
                             record_middle_states=self.record_middle_states, n_middle_states=self.n_middle_states,
                             vis=self.vis)

    def csample_v(self, h, share_random):
        h = h.detach()
        if share_random:
            init_v = func.duplicate(0.5 + torch.randn(*self.lvm.v_shape, device=self.device), h.size(0))
        else:
            init_v = 0.5 + torch.randn(h.size(0), *self.lvm.v_shape, device=self.device)
        init = [init_v]
        return annealed_sgld(fn=functools.partial(self.lvm.energy_net, h=h), init=init, sigma=self.sigma, Ts=self.Ts,
                             denoise=self.denoise, sigma0=self.sigma0, share_random=share_random,
                             record_middle_states=self.record_middle_states, n_middle_states=self.n_middle_states,
                             vis=self.vis)

    def cimpainting_v(self, v, keep, h, share_random):
        h = h.detach()
        init = [v]
        return annealed_sgld_impainting(fn=functools.partial(self.lvm.energy_net, h=h), init=init, keep=keep,
                                        sigma=self.sigma, Ts=self.Ts, denoise=self.denoise, sigma0=self.sigma0,
                                        share_random=share_random, record_middle_states=self.record_middle_states,
                                        n_middle_states=self.n_middle_states, vis=self.vis)
