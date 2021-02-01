import torch
import torch.autograd as autograd
import core.func as func


def R(fn, q: list, p: list, t: float):
    r""" R^t (q, p) = (q, p - t fn'(q))

    """
    with func.RequiresGradContext(*q, requires_grad=True):
        fn_val = fn(*q)
        grads = autograd.grad(fn_val.sum(), q)
    p = [_p - t * _g for _p, _g in zip(p, grads)]
    return q, p


def leap_frog_step(fn, q: list, p: list, m_std: float, h: float):
    r""" One leap frog step
    R^t (q, p) = (q, p - t fn'(q))
    S^t (q, p) = (q + t p / m_std^2, p)

    (q, p) -> R^{h/2} S^h R^{h/2} (q, p)

    Args:
        fn: the energy function
        q: the initial position
        p: the initial momentum
        m_std: the std of momentum
        h: the step size
    """
    m_var = m_std ** 2

    q, p = R(fn, q, p, 0.5 * h)  # R^{h/2}
    q = [_q + h * _p / m_var for _q, _p in zip(q, p)]  # S^h
    q, p = R(fn, q, p, 0.5 * h)  # R^{h/2}
    return q, p


def hamiltonian(fn, q: list, p: list, m_std: float):
    r""" H(q, p)

    Args:
        fn: the energy function
        q: the position
        p: the momentum
        m_std: the std of momentum
    """
    m_var = m_std ** 2
    fn_val = fn(*q).detach()
    n_batch_dim = fn_val.dim()
    K = 0.5 * sum([func.sos(_p, start_dim=n_batch_dim) for _p in p]).detach() / m_var
    return fn_val + K


def hmc_single_step(fn, q: list, m_std: float, h: float, n_leap_frog_steps: int, metropolis_hasting: bool):
    r"""
    Args:
        fn: the energy function
        q: the initial position
        m_std: the std of momentum
        h: the leap frog step size
        n_leap_frog_steps: the number of leap frog steps
        metropolis_hasting: whether to use Metropolis hasting
    """
    p = [m_std * torch.randn_like(_q) for _q in q]
    if metropolis_hasting:
        q_init = q
        H_init = hamiltonian(fn, q, p, m_std)
    for i in range(n_leap_frog_steps):
        q, p = leap_frog_step(fn, q, p, m_std, h)
    if metropolis_hasting:
        H_new = hamiltonian(fn, q, p, m_std)
        prob_accept = (H_init - H_new).exp()
        u = torch.rand_like(prob_accept)
        keep_mask = u > prob_accept
        return [torch.where(func.unsqueeze_like(keep_mask, _q), _q_init, _q) for _q_init, _q in zip(q_init, q)]
    else:
        return q


def hmc(fn, q: list, m_std: float, n_steps: int, h: float, n_leap_frog_steps: int, metropolis_hasting: bool = True):
    r"""
    Args:
        fn: the energy function
        q: the initial position
        m_std: the std of momentum
        n_steps: the number of hmc steps
        h: the leap frog step size
        n_leap_frog_steps: the number of leap frog steps
        metropolis_hasting: whether to use Metropolis hasting
    """
    for i in range(n_steps):
        q = hmc_single_step(fn, q, m_std, h, n_leap_frog_steps, metropolis_hasting)
    return q
