
__all__ = ["NCE"]


from .base import NaiveCriterion
import torch
import torch.nn.functional as F
import numpy as np
import core.func as func
import core.utils.managers as managers


class NCE(NaiveCriterion):
    def __init__(self,
                 nu,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm = models.lvm
        self.c = models.c
        self.nu = nu

    def _sample_from_noise(self, v):
        return torch.randn_like(v, device=v.device)

    def _log_p_noise(self, y):
        return func.log_normal(y, 0., 0., len(self.lvm.v_shape))

    def objective(self, v, **kwargs):
        y = self._sample_from_noise(v)
        log_h_v = F.logsigmoid(-self.lvm.free_energy_net(v) - self.c() - np.log(self.nu) - self._log_p_noise(v))
        log_mh_y = F.logsigmoid(self.lvm.free_energy_net(y) + self.c() + np.log(self.nu) + self._log_p_noise(y))
        return -(log_h_v + self.nu * log_mh_y)
