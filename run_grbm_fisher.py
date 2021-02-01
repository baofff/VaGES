import torch
from interface.datasets.utils import QuickDataset
import torch.autograd as autograd
import logging
from interface.utils import set_seed
from core.utils import global_device
import core.func as func
from core.lvm.rbm import GRBM
from develop.criterions.fisher import VaGESFisher, Fisher
from core.utils import managers
from core.inference.vi.amortized import AmortizedBernoulli
import torch.nn as nn
from core.modules.mlp import TrapezoidMLP
import torch.optim as optim
from interface.runner.fit import naive_fit
from interface.utils.interact import Interact
from core.evaluate import score_on_dataset
import os
from functools import partial
from multiprocessing import Process
from interface.utils.task_schedule import Task
from interface.utils import task_schedule


def score(model, v):
    with func.RequiresGradContext(v, requires_grad=True):
        log_p = -model.free_energy_net(v)
        _score = autograd.grad(log_p.sum(), v, create_graph=True)[0]
    return _score


def grbm_fisher(perturb, dim, batch_size=100, n_its=20000):
    set_seed(1234)
    device = global_device()
    tr, val, te = torch.randn(20000, dim, device=device).split([16000, 2000, 2000])

    grbm = GRBM(v_dim=dim, h_dim=dim, fix_std=True, std=1.)
    grbm.b_v.data.zero_()
    grbm.b_h.data.zero_()
    grbm.W.data.zero_()
    grbm.to(device)
    score_data = score(grbm, te)

    grbm.W.data += perturb * torch.randn_like(grbm.W.data)
    grbm.b_h.data += perturb * torch.randn_like(grbm.b_h.data)
    grbm.b_v.data += perturb * torch.randn_like(grbm.b_v.data)
    score_model = score(grbm, te)
    true_fisher = 0.5 * func.sos(score_data - score_model).mean()

    # models
    critic = TrapezoidMLP(dim, dim, 3, nn.ELU())
    models = managers.ModelsManager(lvm=grbm, critic=critic)

    # optimizers
    optimizers = managers.OptimizersManager(all=optim.Adam(params=critic.parameters(), lr=0.0002))
    estimate = Fisher(models=models, optimizers=optimizers, lr_schedulers=managers.LRSchedulersManager())

    path = "workspace/runner/grbm_fisher/perturb_{}_dim_{}".format(perturb, dim)
    interact = Interact(os.path.join(path, "log"), os.path.join(path, "summary"), 100)
    val_fn = partial(score_on_dataset, score_fn=estimate.objective, batch_size=batch_size)
    naive_fit(estimate, QuickDataset(tr), batch_size=batch_size, n_its=n_its, n_ckpts=10,
              ckpt_root=os.path.join(path, "ckpts"),
              interact=interact, val_dataset=QuickDataset(val), val_fn=val_fn)

    est_fisher = -val_fn(QuickDataset(te))

    logging.info("true fisher: {}, est fisher: {}".format(true_fisher, est_fisher))


def grbm_vages_fisher(perturb, dim, batch_size=100, n_its=20000):
    set_seed(1234)
    device = global_device()
    tr, val, te = torch.randn(20000, dim, device=device).split([16000, 2000, 2000])

    grbm = GRBM(v_dim=dim, h_dim=dim, fix_std=True, std=1.)
    grbm.b_v.data.zero_()
    grbm.b_h.data.zero_()
    grbm.W.data.zero_()
    grbm.to(device)
    score_data = score(grbm, te)

    grbm.W.data += perturb * torch.randn_like(grbm.W.data)
    grbm.b_h.data += perturb * torch.randn_like(grbm.b_h.data)
    grbm.b_v.data += perturb * torch.randn_like(grbm.b_v.data)
    score_model = score(grbm, te)
    true_fisher = 0.5 * func.sos(score_data - score_model).mean()

    # models
    q = AmortizedBernoulli(nn.Linear(dim, dim), dim)
    critic = TrapezoidMLP(dim, dim, 3, nn.ELU())
    models = managers.ModelsManager(lvm=grbm, q=q, critic=critic)

    # optimizers
    optimizers = managers.OptimizersManager(higher=optim.Adam(params=critic.parameters(), lr=0.0002),
                                            lower=optim.Adam(params=q.parameters(), lr=0.0002))
    estimate = VaGESFisher(learn_q=True, n_particles=1, n_lower_steps=5, lower_objective_type="elbo",
                           models=models, optimizers=optimizers, lr_schedulers=managers.LRSchedulersManager())

    path = "workspace/runner/grbm_vages_fisher/perturb_{}_dim_{}".format(perturb, dim)
    interact = Interact(os.path.join(path, "log"), os.path.join(path, "summary"), 100)
    val_fn = partial(score_on_dataset, score_fn=estimate.higher_objective, batch_size=batch_size)
    naive_fit(estimate, QuickDataset(tr), batch_size=batch_size, n_its=n_its, n_ckpts=10,
              ckpt_root=os.path.join(path, "ckpts"),
              interact=interact, val_dataset=QuickDataset(val), val_fn=val_fn)

    est_fisher = -val_fn(QuickDataset(te))

    logging.info("true fisher: {}, est fisher: {}".format(true_fisher, est_fisher))


if __name__ == "__main__":
    target = grbm_vages_fisher  # or grbm_fisher

    tasks = []
    for perturb in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
        for dim in [200, 500]:
            p = Process(target=target, args=(perturb, dim))
            tasks.append(Task(p, 1))
    task_schedule.wait_schedule(tasks, devices=task_schedule.available_devices())
