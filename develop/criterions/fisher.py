from .vages import VaGES
import torch
import torch.autograd as autograd
from core.utils import managers
import core.func as func
from typing import List
from core.criterions.base import NaiveCriterion


class Fisher(NaiveCriterion):
    def __init__(self,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 noise_type: str = 'radermacher',
                 ):
        super().__init__(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers)
        self.lvm, self.critic = models.lvm, models.critic
        self.noise_type = noise_type

    def sample_noise(self, v):
        eps = torch.randn_like(v).to(v.device)
        if self.noise_type == 'radermacher':
            eps = eps.sign()
        elif self.noise_type == 'gaussian':
            pass
        else:
            raise NotImplementedError
        return eps

    def objective(self, v, **kwargs):
        with func.RequiresGradContext(v, requires_grad=True):
            log_p = -self.lvm.free_energy_net(v)
            score = autograd.grad(log_p.sum(), v, create_graph=True)[0]
        eps = self.sample_noise(v)

        with func.RequiresGradContext(v, requires_grad=True):
            critic = self.critic(v)
            grad = autograd.grad((critic * eps).sum(), v, create_graph=True)[0]
            gvp = func.inner_product(grad, eps)

        ip = func.inner_product(score, critic)
        fisher = ip + gvp - 0.5 * func.sos(critic)
        return -fisher

    def update(self, data_loader):
        v = next(data_loader).to(self.device)
        self.models.toggle_grad("critic")
        objective = self.objective(v).mean()
        self.statistics['estimated_fisher'] = -objective.item()
        self.optimizers.all.zero_grad()
        objective.backward()
        self.optimizers.all.step()
        if "all" in self.lr_schedulers:
            self.lr_schedulers.all.step()
        self.record_grad_norm()


class VaGESFisher(VaGES):
    def __init__(self,
                 learn_q: bool,
                 n_particles: int,
                 n_lower_steps: int,
                 lower_objective_type: str,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 noise_type: str = 'radermacher',
                 use_true_post: bool = False,
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
        super().__init__(n_particles=n_particles, n_lower_steps=n_lower_steps, lower_objective_type=lower_objective_type,
                         models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, use_true_post=use_true_post,
                         vr=vr, sgld_steps=sgld_steps, alpha=alpha, sigma=sigma, grad_max_norm=grad_max_norm,
                         sigmas=sigmas, sgld_denoise=sgld_denoise, sgld_sigma0=sgld_sigma0, sgld_type=sgld_type)
        self.lvm, self.q, self.critic = models.lvm, models.q, models.critic
        self.noise_type = noise_type
        self.learn_q = learn_q
        
    def add_noise(self, v):
        return v

    def sample_noise(self, v):
        eps = torch.randn_like(v).to(v.device)
        if self.noise_type == 'radermacher':
            eps = eps.sign()
        elif self.noise_type == 'gaussian':
            pass
        else:
            raise NotImplementedError
        return eps

    def higher_objective(self, v):
        h4z, = self.sample_h_from_post(v, self.n_particles)
        vaes = self.vaes(v, h4z, vr=self.vr)[0].detach()
        eps = self.sample_noise(v)

        with func.RequiresGradContext(v, requires_grad=True):
            critic = self.critic(v)
            grad = autograd.grad((critic * eps).sum(), v, create_graph=True)[0]
            gvp = func.inner_product(grad, eps)

        ip = func.inner_product(vaes, critic)
        fisher = ip + gvp - 0.5 * func.sos(critic)
        return -fisher

    def update(self, data_loader):
        v = next(data_loader).to(self.device)

        if not self.use_true_post and self.learn_q:
            self.models.toggle_grad("q")
            for i in range(self.n_lower_steps):
                lower_objective = self.lower_objective(v).mean()
                self.statistics["%s_lower" % self.lower_objective_type] = lower_objective.item()
                self.optimizers.lower.zero_grad()
                lower_objective.backward()
                self.optimizers.lower.step()
            if "lower" in self.lr_schedulers:
                self.lr_schedulers.lower.step()

        self.models.toggle_grad("critic")
        higher_objective = self.higher_objective(v).mean()
        self.statistics['estimated_fisher'] = -higher_objective.item()
        self.optimizers.higher.zero_grad()
        higher_objective.backward()
        self.optimizers.higher.step()
        if "higher" in self.lr_schedulers:
            self.lr_schedulers.higher.step()

        self.record_grad_norm()
