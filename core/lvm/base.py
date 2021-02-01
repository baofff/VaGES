import torch.nn as nn
from typing import List
import numpy as np


class LVM(nn.Module):
    def __init__(self, v_shape=None, v_dim=None, h_dim=None):
        r""" Latent variable model
        """
        super(LVM, self).__init__()
        self._v_shape = v_shape
        self._v_dim = v_dim
        self._h_dim = h_dim

    @property
    def v_shape(self) -> List[int]:
        if self._v_shape is not None:
            return self._v_shape
        else:
            assert self._v_dim is not None
            return [self._v_dim]

    @property
    def v_dim(self) -> int:
        if self._v_dim is not None:
            return self._v_dim
        else:
            assert self._v_shape is not None
            return int(np.prod(self._v_shape))

    @property
    def h_dim(self) -> int:
        if self._h_dim is not None:
            return self._h_dim
        else:
            raise NotImplementedError

    def cexpect_h(self, v):
        r""" E[h|v]
        Args:
            v: batch_size * v_shape
        """
        raise NotImplementedError

    def cexpect_v(self, h):
        r""" E[v|h]
        Args:
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def csample_h(self, v, n_particles):
        r""" Sample from p(h|v)
        Args:
            v: batch_size * v_shape
            n_particles: #h per v
        """
        raise NotImplementedError

    def csample_v(self, h):
        r""" Sample from p(v|h)
        Args:
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def sample_h(self, n_samples):
        r""" Sample from p(h)
        """
        raise NotImplementedError

    def sample(self, n_samples):
        r""" Sample from p(v)
        """
        raise NotImplementedError

    def energy_net(self, v, h):
        r""" E(v, h)
        Args:
            v: batch_size * v_shape
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def free_energy_net(self, v):
        r""" F(v)
        Args:
            v: batch_size * v_shape
        """
        raise NotImplementedError

    def log_joint(self, v, h):
        r""" log p(v, h)
        Args:
            v: batch_size * v_shape
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def log_cpv(self, v, h):
        r""" log p(v|h)
        Args:
            v: batch_size * v_shape
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def log_cph(self, h, v):
        r""" log p(h|v)
        Args:
            v: batch_size * v_shape
            h: (n_particles *) batch_size * h_dim
        """
        raise NotImplementedError

    def log_likelihood(self, v):
        r""" log p(v)
        Args:
            v: batch_size * v_shape
        """
        raise NotImplementedError

    def energy_net_cut_fn_v(self, v):
        r""" return E(v, ·)
        """
        return lambda _h: self.energy_net(v, _h)
