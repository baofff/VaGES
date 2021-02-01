
__all__ = ["elbo", "iwae", "posterior_fisher"]

import math
import core.func as func
import torch.autograd as autograd


def elbo(v, lvm, q, n_particles=1, eps=None, normalized=True):
    h, log_q = q.implicit_net_log_q(v, n_particles, eps)
    log_p = lvm.log_joint(v, h) if normalized else -lvm.energy_net(v, h)
    return (log_p - log_q).mean(dim=0)


def iwae(v, lvm, q, n_particles, eps=None, normalized=True):
    h, log_q = q.implicit_net_log_q(v, n_particles, eps)
    log_p = lvm.log_joint(v, h) if normalized else -lvm.energy_net(v, h)
    return func.logsumexp(log_p - log_q, dim=0) - math.log(n_particles)


def posterior_fisher(v, lvm, q):
    r""" Fisher divergence between true and variational posteriors
    """
    with func.RequiresGradContext(v, requires_grad=False):
        h = q.implicit_net(v, n_particles=1).squeeze(dim=0)
        log_w = -lvm.energy_net(v, h) - q.log_q(h, v)
        score_w = autograd.grad(log_w.sum(), h, create_graph=True)[0]
        return 0.5 * func.sos(score_w)
