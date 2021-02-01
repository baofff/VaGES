import functools
import core.func as func
import numpy as np


class BruteForceLL(object):
    def __init__(self, lvm, n_particles, h_sampler=None):
        r""" Brute force estimation of log-likelihood by log p(v) = log E_h[p(v|h)]
        Args:
            lvm: a LVM instance
            n_particles: #h per v
            h_sampler: (lvm, n_samples) → samples from p(h) of lvm
        """
        self.lvm = lvm
        self.n_particles = n_particles
        self.h_sampler = h_sampler

    def estimate_ll(self, v):
        r""" estimate the log-likelihood
        Args:
            v: batch_size * v_shape
        """
        h_sampler = self.lvm.sample_h if self.h_sampler is None else functools.partial(self.h_sampler, self.lvm)
        h = h_sampler(self.n_particles * len(v))
        h = h.view(self.n_particles, len(v), *h.shape[1:]).to(v.device)
        log_p = self.lvm.log_cpv(v, h)
        ll = func.logsumexp(log_p, dim=0) - np.log(self.n_particles)
        return ll
