import torch
from tqdm import tqdm
from core.utils import global_device


class GibbsInfo(object):
    def __init__(self, samples, latents, init, middle_states):
        self.samples = samples
        self.latents = latents
        self.init = init
        self.middle_states = middle_states


class Gibbs(object):
    def __init__(self, lvm, n_steps, init_v=None, expectation=True,
                 record_middle_states=False, n_middle_states=None, vis=False):
        """ Gibbs sampler
        Args:
            lvm: a LVM instance
        """
        self.lvm = lvm
        self.n_steps = n_steps
        self.init_v = init_v
        self.expectation = expectation
        self.record_middle_states = record_middle_states
        self.n_middle_states = n_middle_states
        self.vis = vis
        self.device = global_device()

    def sample(self, n_samples):
        v_shape = self.lvm.v_shape if 'v_shape' in self.lvm.__dict__ else [self.lvm.v_dim]
        v = init_v = torch.rand(n_samples, *v_shape) if self.init_v is None else self.init_v
        v = v.to(self.device)
        h = self.lvm.cexpect_h(v).detach()
        n_middle_states = 0 if not self.record_middle_states else self.n_middle_states
        period = self.n_steps // n_middle_states if n_middle_states else self.n_steps + 1
        middle_states = []
        for i in tqdm(range(self.n_steps), desc="Gibbs sampling", disable=not self.vis):
            v = self.lvm.csample_v(h).detach()
            h = self.lvm.csample_h(v, n_particles=1).squeeze(dim=0).detach()
            if (i + 1) % period == 0:
                middle_states.append((i, self.lvm.cexpect_v(h).detach() if self.expectation else v, h))
        v = self.lvm.csample_v(h).detach()
        h = self.lvm.csample_h(v, n_particles=1).squeeze(dim=0).detach()
        return GibbsInfo(samples=self.lvm.cexpect_v(h).detach() if self.expectation else v,
                         latents=h, init=init_v, middle_states=middle_states)
