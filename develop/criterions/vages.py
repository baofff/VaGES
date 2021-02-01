
__all__ = ["VaGES", "VaGESDSM", "VaGESMDSM", "VaGESKSD"]


import torch
import torch.autograd as autograd
import core.inference.vi as vi
from core.inference.sampler.sgld import sgld, sgld_decay, sgld_annealed
from core.criterions.base import Criterion
from core.criterions.sm import make_mdsm_sigmas
from core.utils import managers, diagnose
import core.func as func
from typing import List
from .ksd import rbf_kernel


def score_fn(fn, v, h, avg: bool):
    r""" fn(v, h) & d fn(v, h) / dv

    Args:
        fn: batch_size * v_shape, n_particles * batch_size * h_dim → batch_size
        v: batch_size * v_shape
        h: n_particles * batch_size * h_dim
        avg: whether to average over 'n_particles' dimension of h
    """
    assert v.size(0) == h.size(1)
    if avg:
        with func.RequiresGradContext(v, requires_grad=True):
            fn_val = fn(v, h).mean(dim=0)
            score = autograd.grad(fn_val.sum(), v, create_graph=True)[0]
        return score, fn_val
    else:
        batch_sizes, n_particles = h.size(0), h.size(1)
        v = func.duplicate(v, n_particles).flatten(0, 1)
        h = h.flatten(0, 1)
        with func.RequiresGradContext(v, requires_grad=True):
            fn_val = fn(v, h)
            assert fn_val.dim() == 1 and fn_val.size(0) == batch_sizes * n_particles
            score = autograd.grad(fn_val.sum(), v, create_graph=True)[0]
        fn_val = fn_val.view(n_particles, batch_sizes)
        score = score.view(n_particles, batch_sizes, *score.shape[1:])
        return score, fn_val


class VaGES(Criterion):
    def __init__(self,
                 n_particles: int,
                 n_lower_steps: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 biased_cov: bool = False,
                 cp_sd_cov: bool = False,
                 use_true_post: bool = False,
                 no_cov: bool = False,
                 vr: bool = False,
                 # for sgld
                 sgld_steps: int = 0,
                 alpha: float = 4e-4,
                 sigma: float = 0.01,
                 grad_max_norm: float = None,
                 # for sgld_annealed
                 sigmas: List[float] = None,
                 sgld_denoise: bool = None,
                 sgld_sigma0: float = None,
                 sgld_type: str = "default"
                 ):
        assert lower_objective_type in ["elbo", "posterior_fisher"]
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm, self.q = models.lvm, models.q
        self.n_particles = n_particles
        self.n_lower_steps = n_lower_steps
        self.lower_objective_type = lower_objective_type
        self.biased_cov = biased_cov
        self.cp_sd_cov = cp_sd_cov
        self.use_true_post = use_true_post
        self.no_cov = no_cov
        self.vr = vr
        # sgld hyper-parameters
        self.sgld_steps = sgld_steps
        self.alpha = alpha
        self.sigma = sigma
        self.grad_max_norm = grad_max_norm
        # sgld_annealed
        self.sigmas = sigmas
        self.sgld_denoise = sgld_denoise
        self.sgld_sigma0 = sgld_sigma0
        self.sgld_type = sgld_type

    def sample_h_from_post(self, v, *n_particles):
        n_particles_sum = sum(n_particles)
        if self.use_true_post:
            h = self.lvm.csample_h(v, n_particles_sum).detach()
        else:
            h = self.q.implicit_net(v, n_particles_sum).detach()

            if self.sgld_type is not None:
                fn = self.lvm.energy_net_cut_fn_v(v)
                if self.sgld_type == "default":
                    h = sgld(fn=fn, init=[h], alpha=self.alpha, sigma=self.sigma, n_steps=self.sgld_steps,
                             grad_max_norms=self.grad_max_norm, vis=False)[0]
                elif self.sgld_type == "decay":
                    h = sgld_decay(fn=fn, init=[h], init_alpha=self.alpha, n_steps=self.sgld_steps,
                                   grad_max_norms=self.grad_max_norm, vis=False)[0]
                elif self.sgld_type == "annealed":
                    h = sgld_annealed(fn=fn, init=[h], alpha=self.alpha, sigmas=self.sigmas,
                                      denoise=self.sgld_denoise, sigma0=self.sgld_sigma0,
                                      grad_max_norms=self.grad_max_norm, vis=False)[0]
                else:
                    raise ValueError
        return h.split(n_particles, dim=0)

    def score_joint(self, v, h, avg=False):
        r""" d log p(v, h) / dv

        Args:
            v: batch_size * v_shape
            h: n_particles * batch_size * h_dim
            avg: whether to average over 'n_particles' dimension of h
        """
        fn = lambda vv, hh: -self.lvm.energy_net(vv, hh)
        return score_fn(fn, v, h, avg)

    def score_w(self, v, h, avg=False):
        r""" d log [p(v, h) / q(h|v)] / dv

        Args:
            v: batch_size * v_shape
            h: n_particles * batch_size * h_dim
            avg: whether to average over 'n_particles' dimension of h
        """
        if self.use_true_post:
            fn = lambda vv, hh: -self.lvm.energy_net(vv, hh) - self.lvm.log_cph(hh, vv)
        else:
            fn = lambda vv, hh: -self.lvm.energy_net(vv, hh) - self.q.log_q(hh, vv)
        return score_fn(fn, v, h, avg)

    def vaes(self, v, h, vr=False):
        return self.score_w(v, h, avg=True) if vr else self.score_joint(v, h, avg=True)

    def before_esd(self, v, h, z):
        r""" For calculating the expectation of the second derivative

        Args:
            v: batch_size * v_shape
            h: n_particles * batch_size * h_dim
            z: batch_size * v_shape
        """
        score_joint = self.score_joint(v, h, avg=True)[0]
        return func.inner_product(z, score_joint)

    def before_cov(self, v, h, z):
        r""" For calculating the covariance

        Args:
            v: batch_size * v_shape
            h: n_particles * batch_size * h_dim
            z: batch_size * v_shape
        """
        with func.RequiresGradContext(v, self.lvm, requires_grad=[True, False]):
            log_p_1 = -self.lvm.energy_net(v, h)  # log p(v, h; theta_d)
        with func.RequiresGradContext(v, self.lvm, requires_grad=[False, True]):
            log_p_2 = -self.lvm.energy_net(v, h)  # log p(v_d, h; theta)
        with func.RequiresGradContext(v, self.lvm, requires_grad=[True, True]):
            fn_val = (log_p_1 * log_p_2).mean(dim=0)
            _score_fn = autograd.grad(fn_val.sum(), v, create_graph=True)[0]
        coupled = func.inner_product(z, _score_fn)

        with func.RequiresGradContext(v, requires_grad=True):
            score_u = autograd.grad(log_p_1.mean(dim=0).sum(), v, retain_graph=True)[0].detach()
        ip_u = func.inner_product(z, score_u)
        log_p_u = log_p_2.mean(dim=0)
        uncoupled = ip_u * log_p_u

        res = coupled - uncoupled
        if not self.biased_cov:
            res = self.n_particles / (self.n_particles-1) * res
        return res

    def before_vages(self, v, h4esd, h4cov, z):
        if self.no_cov:
            return self.before_esd(v, h4esd, z)
        else:
            return self.before_esd(v, h4esd, z) + self.before_cov(v, h4cov, z)

    def add_noise(self, v):
        raise NotImplementedError

    def lower_objective(self, v):
        v_noised = self.add_noise(v)
        if self.lower_objective_type == "elbo":
            return -vi.elbo(v_noised, self.lvm, self.q, normalized=False)
        elif self.lower_objective_type == "posterior_fisher":
            return vi.posterior_fisher(v_noised, self.lvm, self.q)
        else:
            raise NotImplementedError

    def higher_objective(self, v):
        raise NotImplementedError

    def update(self, data_loader):
        v = next(data_loader).to(self.device)

        if not self.use_true_post:
            self.models.toggle_grad("q")
            for i in range(self.n_lower_steps):
                lower_objective = self.lower_objective(v).mean()
                self.statistics["%s_lower" % self.lower_objective_type] = lower_objective.item()
                self.optimizers.lower.zero_grad()
                lower_objective.backward()
                self.optimizers.lower.step()
            if "lower" in self.lr_schedulers:
                self.lr_schedulers.lower.step()

        self.models.toggle_grad("lvm")
        higher_objective = self.higher_objective(v).mean()
        self.statistics['%s_higher' % self.criterion_name()] = higher_objective.item()
        self.optimizers.higher.zero_grad()
        higher_objective.backward()
        self.optimizers.higher.step()
        if "higher" in self.lr_schedulers:
            self.lr_schedulers.higher.step()

        self.record_grad_norm()


class VaGESDSM(VaGES):
    def __init__(self,
                 noise_std,
                 n_particles: int,
                 n_lower_steps: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 biased_cov: bool = False,
                 cp_sd_cov: bool = False,
                 use_true_post: bool = False,
                 no_cov: bool = False,
                 vr: bool = False,
                 sgld_steps: int = 0,
                 alpha: float = 2.,
                 sigma: float = 0.01,
                 grad_max_norm: float = None,
                 sigmas: List[float] = None,
                 sgld_denoise: bool = None,
                 sgld_sigma0: float = None,
                 sgld_type: str = "default"
                 ):
        super().__init__(n_particles, n_lower_steps, lower_objective_type, models, optimizers, lr_schedulers,
                         biased_cov, cp_sd_cov, use_true_post, no_cov, vr, sgld_steps, alpha, sigma, grad_max_norm,
                         sigmas, sgld_denoise, sgld_sigma0, sgld_type)
        self.noise_std = noise_std

    def add_noise(self, v):
        eps = torch.randn_like(v, device=v.device)
        return v + self.noise_std * eps

    def higher_objective(self, v):
        eps = torch.randn_like(v, device=v.device)
        v_noised = v + self.noise_std * eps
        score_parzen = -eps / self.noise_std
        h4z, h4esd, h4cov = self.sample_h_from_post(v_noised, *[self.n_particles] * 3)
        if self.cp_sd_cov:
            h4cov = h4esd
        vaes = self.vaes(v_noised, h4z, vr=self.vr)[0].detach()
        z = (vaes - score_parzen).detach()
        del vaes, score_parzen
        return self.before_vages(v_noised, h4esd, h4cov, z)


class VaGESMDSM(VaGES):
    def __init__(self,
                 sigma0, sigma_begin, sigma_end, dist,
                 n_particles: int,
                 n_lower_steps: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 biased_cov: bool = False,
                 cp_sd_cov: bool = False,
                 use_true_post: bool = False,
                 no_cov: bool = False,
                 vr: bool = False,
                 sgld_steps: int = 0,
                 alpha: float = 2.,
                 sigma: float = 0.01,
                 grad_max_norm: float = None,
                 sigmas: List[float] = None,
                 sgld_denoise: bool = None,
                 sgld_sigma0: float = None,
                 sgld_type: str = "default"
                 ):
        super().__init__(n_particles, n_lower_steps, lower_objective_type, models, optimizers, lr_schedulers,
                         biased_cov, cp_sd_cov, use_true_post, no_cov, vr, sgld_steps, alpha, sigma, grad_max_norm,
                         sigmas, sgld_denoise, sgld_sigma0, sgld_type)
        self.sigma0 = sigma0
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        self.dist = dist

    def randomness(self, v):
        r""" The randomness of mdsm includes:
            sigmas: the random noise level
            eps: the standard gauss noise
        """
        sigmas = make_mdsm_sigmas(v.size(0), self.sigma_begin, self.sigma_end, self.dist, device=v.device)
        eps = torch.randn_like(v, device=v.device)
        return sigmas, eps

    def add_noise(self, v):
        sigmas, eps = self.randomness(v)
        sigmas4v = sigmas.view(len(v), *([1] * (v.dim() - 1)))
        return v + sigmas4v * eps

    def higher_objective(self, v):
        sigmas, eps = self.randomness(v)
        sigmas4v = sigmas.view(len(v), *([1] * (v.dim() - 1)))
        v_noised = v + sigmas4v * eps
        h4z, h4esd, h4cov = self.sample_h_from_post(v_noised, *[self.n_particles] * 3)
        if self.cp_sd_cov:
            h4cov = h4esd
        vaes = self.vaes(v_noised, h4z, vr=self.vr)[0].detach()
        z = ((vaes / sigmas4v + eps / self.sigma0 ** 2) / sigmas4v).detach()
        del vaes, eps
        return self.before_vages(v_noised, h4esd, h4cov, z)

    def lower_objective(self, v):
        sigmas = make_mdsm_sigmas(v.size(0), self.sigma_begin, self.sigma_end, self.dist, device=v.device)
        return super().lower_objective(v) / sigmas


class VaGESKSD(VaGES):
    def __init__(self,
                 n_particles: int,
                 n_lower_steps: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 biased_cov: bool = False,
                 cp_sd_cov: bool = False,
                 use_true_post: bool = False,
                 no_cov: bool = False,
                 vr: bool = False,
                 # for sgld
                 sgld_steps: int = 0,
                 alpha: float = 4e-4,
                 sigma: float = 0.01,
                 grad_max_norm: float = None,
                 # for sgld_annealed
                 sigmas: List[float] = None,
                 sgld_denoise: bool = None,
                 sgld_sigma0: float = None,
                 sgld_type: str = "default"
                 ):
        super().__init__(n_particles, n_lower_steps, lower_objective_type, models, optimizers, lr_schedulers,
                         biased_cov, cp_sd_cov, use_true_post, no_cov, vr, sgld_steps, alpha, sigma, grad_max_norm,
                         sigmas, sgld_denoise, sgld_sigma0, sgld_type)

    def add_noise(self, v):
        return v

    def higher_objective(self, v):
        h4z, h4esd, h4cov = self.sample_h_from_post(v, *[self.n_particles] * 3)
        bs = v.shape[0]
        assert bs % 2 == 0
        v, vp = v[bs // 2:], v[: bs // 2]
        vaes = self.vaes(v, h4z[:, bs // 2:], vr=self.vr)[0].detach()
        ker, grad_ker = rbf_kernel(v, vp)
        z = (ker * vaes + grad_ker).detach()
        return self.before_vages(vp, h4esd[:, : bs // 2], h4cov[:, : bs // 2], z)
