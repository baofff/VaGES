import os
from core.evaluate import grid_sample
from core.inference.sampler.gibbs import Gibbs
from .base import Evaluator
import functools
from interface.utils.interact import Interact
from interface.datasets import DatasetFactory
from core.utils.managers import ModelsManager
from core.evaluate import score_on_dataset
from core.criterions import ssm
from core.inference.ll import AIS4GRBM
import interface.utils.plot as plot
import torch
from core.utils import device_of


class RBMEvaluator(Evaluator):
    def __init__(self, models: ModelsManager, evaluator_root: str, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Evaluate RBM
        Args:
            models: an object of ModelsManager
            evaluator_root: the root to put evaluation results
            options: a dict, evaluation function name -> arguments of the function
                Example: {"grid_sample": {"nrow": 10, "ncol": 10}}
            dataset: an instance of DatasetFactory
            interact: an instance of Interact
        """
        super().__init__(evaluator_root, options)
        self.models = models
        self.lvm = models.lvm
        self.dataset = dataset
        self.unpreprocess_fn = None if self.dataset is None else self.dataset.unpreprocess
        self.interact = interact

    def sample_fn(self, n_samples, n_steps, expectation=True):
        sampler = Gibbs(self.lvm, n_steps, expectation=expectation)
        return sampler.sample(n_samples).samples

    def grid_sample(self, it, nrow=10, ncol=10, n_steps=1000):
        fname = os.path.join(self.evaluator_root, "grid_sample", "%d.png" % it)
        grid_sample(fname, nrow, ncol, functools.partial(self.sample_fn, n_steps=n_steps), self.unpreprocess_fn)

    def ssm_score(self, it, batch_size=None):
        test_dataset = self.dataset.get_test_data()
        if batch_size is None:
            batch_size = len(test_dataset)
        ssm_score = score_on_dataset(test_dataset, score_fn=functools.partial(ssm, lvm=self.lvm), batch_size=batch_size)
        self.interact.report_scalar(ssm_score, it, "ssm_score")

    def log_likelihood_score(self, it, batch_size=None):
        test_dataset = self.dataset.get_test_data()
        if batch_size is None:
            batch_size = len(test_dataset)
        ais = AIS4GRBM(self.lvm)
        ais.update_log_partition()
        ll_score = score_on_dataset(test_dataset, score_fn=ais.estimate_ll, batch_size=batch_size)
        self.interact.report_scalar(ll_score, it, "ll_score")

    def plot_sample_density(self, it, left, right):
        assert self.lvm.v_dim == 2
        fname = os.path.join(self.evaluator_root, "plot_sample_density", "%d.png" % it)
        xs = torch.linspace(left, right, steps=100)
        xs, ys = torch.meshgrid([xs, xs])
        xs, ys = xs.flatten().unsqueeze(dim=-1), ys.flatten().unsqueeze(dim=-1)
        v = self.dataset.preprocess(torch.cat([xs, ys], dim=-1).to(device_of(self.lvm)))
        density = -self.lvm.free_energy_net(v)
        density = (density - density.max()).exp()
        xs, ys, = xs.view(100, 100).detach().cpu().numpy(), ys.view(100, 100).detach().cpu().numpy()
        density = density.view(100, 100).detach().cpu().numpy()
        plot.plot_density(xs, ys, density, fname)

    def plot_sample_scatter(self, it, n_steps=1000):
        assert self.lvm.v_dim == 2
        fname = os.path.join(self.evaluator_root, "plot_sample_scatter", "%d.png" % it)
        samples = self.sample_fn(1000, n_steps, expectation=False)
        samples = self.unpreprocess_fn(samples).detach().cpu().numpy()
        plot.plot_scatter(samples, fname)
