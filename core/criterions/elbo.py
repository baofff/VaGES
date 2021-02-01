
__all__ = ["ELBO", "IWAE"]


from .base import NaiveCriterion
import core.inference.vi as vi
import core.utils.managers as managers


class ELBO(NaiveCriterion):
    def __init__(self,
                 n_particles: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        r"""
        Args:
            n_particles: number of particles in Monte Carlo estimate
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager, containing only one optimizer indexed by 'all'
            lr_schedulers: an object of LRSchedulersManager, containing only one scheduler indexed by 'all'
        """
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm, self.q = models.lvm, models.q
        self.n_particles = n_particles

    def criterion_name(self):
        return "ELBO%d" % self.n_particles

    def objective(self, v, **kwargs):
        return -vi.elbo(v, self.lvm, self.q, self.n_particles)


class IWAE(NaiveCriterion):
    def __init__(self,
                 n_particles: int,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        r"""
        Args:
            n_particles: number of particles in Monte Carlo estimate
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager, containing only one optimizer indexed by 'all'
            lr_schedulers: an object of LRSchedulersManager, containing only one scheduler indexed by 'all'
        """
        super().__init__(models, optimizers, lr_schedulers)
        self.lvm, self.q = models.lvm, models.q
        self.n_particles = n_particles

    def criterion_name(self):
        return "IWAE%d" % self.n_particles

    def objective(self, v, **kwargs):
        return -vi.iwae(v, self.lvm, self.q, self.n_particles)
