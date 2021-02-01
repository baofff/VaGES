
__all__ = ["BiSSM", "BiDSM", "BiMDSM"]


import torch
import torch.autograd as autograd
import core.inference.vi as vi
from .base import Criterion
from .sm import make_mdsm_sigmas, make_ssm_noise
import higher
from core.utils import managers, diagnose
import core.func as func


class BiSM(Criterion):
    def __init__(self,
                 n_lower_steps: int,
                 n_unroll: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 ):
        assert lower_objective_type in ["elbo", "posterior_fisher"]
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm, self.q = models.lvm, models.q
        self.n_lower_steps = n_lower_steps
        self.n_unroll = n_unroll
        self.lower_objective_type = lower_objective_type

    def criterion_name(self):
        return "{}{}".format(self.__class__.__name__.lower(), self.n_unroll)

    def add_noise(self, v):
        raise NotImplementedError

    def default_val_fn(self, v):
        return self.higher_objective(v)

    def lower_objective(self, v, q=None):
        if q is None:
            q = self.q
        v_noised = self.add_noise(v)
        if self.lower_objective_type == "elbo":
            return -vi.elbo(v_noised, self.lvm, q, normalized=False)
        elif self.lower_objective_type == "posterior_fisher":
            return vi.posterior_fisher(v_noised, self.lvm, q)
        else:
            raise NotImplementedError

    def higher_objective(self, v, q=None):
        raise NotImplementedError

    def update(self, data_loader):
        v = next(data_loader).to(self.device)

        # lower level optimization
        self.models.toggle_grad("q")
        for i in range(self.n_lower_steps):
            lower_objective = self.lower_objective(v).mean()
            self.statistics["%s_lower" % self.lower_objective_type] = lower_objective.item()
            self.optimizers.lower.zero_grad()
            lower_objective.backward()
            self.optimizers.lower.step()
        if "lower" in self.lr_schedulers:
            self.lr_schedulers.lower.step()

        # use unroll to solve the lower level optimization
        self.models.toggle_grad("lvm", "q")
        with higher.innerloop_ctx(self.q, self.optimizers.lower) as (fq, diffopt_q):
            for i in range(self.n_unroll):
                inner_loss = self.lower_objective(v, q=fq).mean()
                diffopt_q.step(inner_loss)  # phi^n-1(theta) -> phi^n(theta)
            higher_objective = self.higher_objective(v, q=fq).mean()
            self.statistics['%s_higher' % self.criterion_name()] = higher_objective.item()
            self.models.toggle_grad("lvm")
            self.optimizers.higher.zero_grad()
            higher_objective.backward()
            self.optimizers.higher.step()
            if "higher" in self.lr_schedulers:
                self.lr_schedulers.higher.step()

        self.record_grad_norm()


class BiSSM(BiSM):
    def __init__(self,
                 n_lower_steps: int,
                 n_unroll: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 noise_type='radermacher'
                 ):
        super().__init__(n_lower_steps, n_unroll, lower_objective_type, models, optimizers, lr_schedulers)
        self.noise_type = noise_type

    def add_noise(self, v):
        return v

    def _sample(self, v):
        u = torch.randn_like(v).to(v.device)
        if self.noise_type == 'radermacher':  # better
            u = u.sign()
        elif self.noise_type == 'gaussian':
            pass
        else:
            raise NotImplementedError
        return u, v.clone().detach()

    def higher_objective(self, v, q=None):
        if q is None:
            q = self.q
        u = make_ssm_noise(*v.shape, noise_type=self.noise_type, device=v.device)
        h = q.implicit_net(v.detach(), n_particles=1).squeeze(dim=0)
        with func.RequiresGradContext(v, requires_grad=True):
            log_w = -self.lvm.energy_net(v, h) - q.log_q(h, v)
            score_w = autograd.grad(log_w.sum(), v, create_graph=True)[0]
            loss1 = 0.5 * func.sos(score_w)
            hvp = autograd.grad((score_w * u).sum(), v, create_graph=True)[0]
            loss2 = func.inner_product(hvp, u)
        return loss1 + loss2


class BiDSM(BiSM):
    def __init__(self,
                 noise_std,
                 n_lower_steps: int,
                 n_unroll: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 n_particles=1):
        super().__init__(n_lower_steps, n_unroll, lower_objective_type, models, optimizers, lr_schedulers)
        self.noise_std = noise_std
        self.n_particles = n_particles

    def add_noise(self, v):
        eps = torch.randn_like(v, device=v.device)
        return v + self.noise_std * eps

    def higher_objective(self, v, q=None):
        if q is None:
            q = self.q
        eps = torch.randn_like(v, device=v.device)
        v_noised = v + self.noise_std * eps
        score_parzen = -eps / self.noise_std

        h = q.implicit_net(v_noised.detach(), self.n_particles).flatten(0, 1)
        v_noised = func.duplicate(v_noised, self.n_particles).flatten(0, 1)
        score_parzen = func.duplicate(score_parzen, self.n_particles).flatten(0, 1)

        with func.RequiresGradContext(v_noised, requires_grad=True):
            log_w = -self.lvm.energy_net(v_noised, h) - q.log_q(h, v_noised)
            score = autograd.grad(log_w.sum(), v_noised, create_graph=True)[0]

        return 0.5 * func.sos(score - score_parzen).view(self.n_particles, v.size(0)).mean(dim=0)


class BiMDSM(BiSM):
    def __init__(self,
                 sigma0, sigma_begin, sigma_end, dist,
                 n_lower_steps: int,
                 n_unroll: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,):
        super().__init__(n_lower_steps, n_unroll, lower_objective_type, models, optimizers, lr_schedulers)
        self.sigma0 = sigma0
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        self.dist = dist
        self.inner_loss_div_sigmas = True

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

    def higher_objective(self, v, q=None):
        if q is None:
            q = self.q
        sigmas, eps = self.randomness(v)
        sigmas4v = sigmas.view(len(v), *([1] * (v.dim() - 1)))
        v_noised = v + sigmas4v * eps

        h = q.implicit_net(v_noised.detach(), n_particles=1).squeeze(dim=0)
        with func.RequiresGradContext(v_noised, requires_grad=True):
            log_w = -self.lvm.energy_net(v_noised, h) - q.log_q(h, v_noised)
            score_w = autograd.grad(log_w.sum(), v_noised, create_graph=True)[0]
        return 0.5 * func.sos(score_w / sigmas4v + eps / self.sigma0 ** 2)

    def lower_objective(self, v, q=None):
        sigmas = make_mdsm_sigmas(v.size(0), self.sigma_begin, self.sigma_end, self.dist, device=v.device)
        return super().lower_objective(v, q=q) / sigmas
