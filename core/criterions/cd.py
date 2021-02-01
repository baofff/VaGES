
__all__ = ["CD", "PCD"]


from .base import NaiveCriterion
import core.utils.managers as managers
from core.evaluate import reconstruct_error


class CD(NaiveCriterion):
    def __init__(self,
                 k: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager):
        super(CD, self).__init__(models, optimizers, lr_schedulers)
        r"""
        Args:
            k: number of steps in Gibbs sampling
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager, containing only one optimizer indexed by 'all'
            lr_schedulers: an object of LRSchedulersManager, containing only one scheduler indexed by 'all'
        """
        self.lvm = models.lvm
        self.k = k

    def criterion_name(self):
        return "cd%d" % self.k

    def default_val_fn(self, v):
        return reconstruct_error(self.models, v)

    def objective(self, v, **kwargs):
        h = self.lvm.csample_h(v, n_particles=1).squeeze(dim=0).detach()
        e1 = self.lvm.free_energy_net(v)
        h_sample = h
        for _ in range(self.k):
            v_sample = self.lvm.csample_v(h_sample).detach()
            h_sample = self.lvm.csample_h(v_sample, n_particles=1).squeeze(dim=0).detach()
        e2 = self.lvm.free_energy_net(v_sample)
        return (e1 - e2).mean()


class PCD(NaiveCriterion):
    def __init__(self,
                 k: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager):
        super(PCD, self).__init__(models, optimizers, lr_schedulers)
        r"""
        Args:
            k: number of steps in Gibbs sampling
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager, containing only one optimizer indexed by 'all'
            lr_schedulers: an object of LRSchedulersManager, containing only one scheduler indexed by 'all'
        """
        self.lvm = models.lvm
        self.k = k
        self.persistent = None  # todo: it requires a state dict

    def criterion_name(self):
        return "pcd%d" % self.k

    def default_val_fn(self, v):
        return reconstruct_error(self.models, v)

    def objective(self, v, **kwargs):
        if self.persistent is None:
            self.persistent = self.lvm.csample_h(v, n_particles=1).squeeze(dim=0).detach()
        e1 = self.lvm.free_energy_net(v)
        h_sample = self.persistent
        for _ in range(self.k):
            v_sample = self.lvm.csample_v(h_sample).detach()
            h_sample = self.lvm.csample_h(v_sample, n_particles=1).squeeze(dim=0).detach()
        e2 = self.lvm.free_energy_net(v_sample)
        self.persistent = h_sample
        return (e1 - e2).mean()
