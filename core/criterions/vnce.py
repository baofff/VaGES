
__all__ = ["VNCE"]


from .base import Criterion
import torch
import torch.nn.functional as F
import numpy as np
import core.func as func
import core.inference.vi as vi
import core.utils.managers as managers


class VNCE(Criterion):
    def __init__(self,
                 nu,
                 n_particles,
                 n_lower_steps: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm, self.q, self.c = models.lvm, models.q, models.c
        self.nu = nu
        self.n_particles = n_particles
        self.n_lower_steps = n_lower_steps

    def _sample_from_noise(self, v):
        return torch.randn_like(v, device=v.device)

    def _log_p_noise(self, y):
        return func.log_normal(y, 0., 0., len(self.lvm.v_shape))

    def higher_objective(self, v):
        hv, log_qv = self.q.implicit_net_log_q(v, n_particles=1)
        hv, log_qv = hv.squeeze(dim=0), log_qv.squeeze(dim=0)
        y = self._sample_from_noise(v)
        hy, log_qy = self.q.implicit_net_log_q(y, n_particles=self.n_particles)
        log_py = -self.lvm.energy_net(y, hy)
        log_wy = log_py - log_qy
        iwae = func.logsumexp(log_wy, dim=0) - np.log(self.n_particles)

        log_h_v = F.logsigmoid(-self.lvm.energy_net(v, hv) - log_qv - self.c() - np.log(self.nu) - self._log_p_noise(v))
        log_mh_y = F.logsigmoid(-iwae + self.c() + np.log(self.nu) + self._log_p_noise(y))
        return -(log_h_v + self.nu * log_mh_y)

    def lower_objective(self, v):
        return -vi.elbo(v, self.lvm, self.q, normalized=False)

    def update(self, data_loader):
        v = next(data_loader).to(self.device)

        # update q
        self.models.toggle_grad("q")
        for i in range(self.n_lower_steps):
            lower_objective = self.lower_objective(v).mean()
            self.statistics["elbo_lower"] = lower_objective.item()
            self.optimizers.lower.zero_grad()
            lower_objective.backward()
            self.optimizers.lower.step()
        if "lower" in self.lr_schedulers:
            self.lr_schedulers.lower.step()

        # update lvm and c
        self.models.toggle_grad("lvm", "c")
        higher_objective = self.higher_objective(v).mean()
        self.statistics["vnce_higher"] = higher_objective.item()
        self.optimizers.higher.zero_grad()
        higher_objective.backward()
        self.optimizers.higher.step()
        if "higher" in self.lr_schedulers:
            self.lr_schedulers.higher.step()
