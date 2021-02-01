import os
from core.evaluate import grid_sample, sample2dir, variational_posterior_extract_feature,\
    plot_variational_posterior_embedding
from core.inference.sampler.sgld import GeometricAnnealedSGLD
from .base import Evaluator
import functools
from interface.utils.interact import Interact
from interface.datasets import DatasetFactory
from interface.utils.misc import sample_from_dataset
from core.utils.managers import ModelsManager
from core.utils import global_device
import numpy as np
import torch
from .utils import linear_svm_classify, linear_interpolate, rect_interpolate
import interface.utils.plot as plot
import random
from torchvision.utils import make_grid, save_image
from core import func
from interface.datasets import Mnist


class EBLVMEvaluator(Evaluator):
    def __init__(self, models: ModelsManager, evaluator_root: str, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Evaluate EBLVM
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
        if 'q' in models:
            self.q = models.q
        self.dataset = dataset
        self.train_dataset = dataset.get_train_data()
        self.unpreprocess_fn = None if self.dataset is None else self.dataset.unpreprocess
        self.interact = interact
        self.device = global_device()

    def sample_fn(self, n_samples, sigma, steps_max, steps_annealed, steps_min, Tmax, Tmin, denoise, sigma0, use_q):
        sampler = GeometricAnnealedSGLD(self.lvm, sigma=sigma,
                                        steps_max=steps_max, steps_annealed=steps_annealed, steps_min=steps_min,
                                        Tmax=Tmax, Tmin=Tmin, denoise=denoise, sigma0=sigma0)
        if use_q:
            idxes = np.random.randint(0, len(self.train_dataset), size=n_samples)
            v = torch.stack(list(map(lambda idx: self.train_dataset[idx], idxes)), dim=0).to(self.device)
            feature = self.q.expect(v)
            return sampler.csample_v(feature, share_random=False).samples[0]
        else:
            return sampler.sample_joint(n_samples).samples[0]

    def grid_sample(self, it, nrow=10, ncol=10, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                    Tmax=100, Tmin=1, denoise=True, sigma0=0.1, use_q=True):
        fname = os.path.join(self.evaluator_root, "grid_sample", "%d.png" % it)
        sample_fn = functools.partial(self.sample_fn, sigma=sigma, steps_max=steps_max, steps_annealed=steps_annealed,
                                      steps_min=steps_min, Tmax=Tmax, Tmin=Tmin,
                                      denoise=denoise, sigma0=sigma0, use_q=use_q)
        grid_sample(fname, nrow, ncol, sample_fn, self.unpreprocess_fn)

    def sample2dir(self, n_samples, batch_size, path=None, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                   Tmax=100, Tmin=1, denoise=True, sigma0=0.1, use_q=True, persist=True, it=None):
        if path is None:
            path = os.path.join(self.evaluator_root, "sample2dir")
        sample_fn = functools.partial(self.sample_fn, sigma=sigma, steps_max=steps_max, steps_annealed=steps_annealed,
                                      steps_min=steps_min, Tmax=Tmax, Tmin=Tmin,
                                      denoise=denoise, sigma0=sigma0, use_q=use_q)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)

    def plot_variational_posterior_embedding(self, it, batch_size):
        fname = os.path.join(self.evaluator_root, "plot_variational_posterior_embedding", "%d.png" % it)
        labelled = self.dataset.allow_labelled()
        plot_variational_posterior_embedding(fname, self.q, self.dataset.get_test_data(labelled=labelled), labelled,
                                             batch_size)

    def variational_posterior_embedding_classify(self, it, batch_size):
        train_features, train_labels = variational_posterior_extract_feature(self.q,
                                                                             self.dataset.get_test_data(labelled=True),
                                                                             True, batch_size)
        test_features, test_labels = variational_posterior_extract_feature(self.q,
                                                                           self.dataset.get_test_data(labelled=True),
                                                                           True, batch_size)
        acc = linear_svm_classify(train_features, train_labels, test_features, test_labels)[-1]
        self.interact.report_scalar(acc, it, "linear_svm_classify_acc")

    def plot_sample_scatter(self, it, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                            Tmax=100, Tmin=1, denoise=True, sigma0=0.1, use_q=True):
        assert self.lvm.v_dim == 2
        fname = os.path.join(self.evaluator_root, "plot_sample_scatter", "%d.png" % it)
        samples = self.sample_fn(1000, sigma=sigma, steps_max=steps_max, steps_annealed=steps_annealed,
                                 steps_min=steps_min, Tmax=Tmax, Tmin=Tmin,
                                 denoise=denoise, sigma0=sigma0, use_q=use_q)
        samples = self.unpreprocess_fn(samples).detach().cpu().numpy()
        plot.plot_scatter(samples, fname)

    def deterministic_interpolate_rect(self, it, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                                       Tmax=100, Tmin=1, denoise=True, sigma0=0.1):
        fname = os.path.join(self.evaluator_root, "deterministic_interpolate_rect", "%d.png" % it)
        os.makedirs(os.path.join(self.evaluator_root, "deterministic_interpolate_rect"), exist_ok=True)
        idxes = random.sample(range(len(self.train_dataset)), 3)
        samples = list(map(lambda idx: self.train_dataset[idx], idxes))
        v = torch.stack(samples, dim=0).to(self.device)
        feature = self.q.expect(v)
        h = rect_interpolate(*feature, steps=10).to(self.device)

        sampler = GeometricAnnealedSGLD(self.lvm, sigma=sigma,
                                        steps_max=steps_max, steps_annealed=steps_annealed, steps_min=steps_min,
                                        Tmax=Tmax, Tmin=Tmin, denoise=denoise, sigma0=sigma0)
        samples = sampler.csample_v(h, share_random=True).samples[0]
        samples = self.unpreprocess_fn(samples)
        grid = make_grid(samples, 10)
        save_image(grid, fname)

    def deterministic_interpolate_linear(self, it, n=10, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                                         Tmax=100, Tmin=1, denoise=True, sigma0=0.1, ratio=3., record_middle_states=True,
                                         n_middle_states=10, n_interpolation=10):
        os.makedirs(os.path.join(self.evaluator_root, "deterministic_interpolate_linear"), exist_ok=True)
        sampler = GeometricAnnealedSGLD(self.lvm, sigma=sigma, steps_max=steps_max, steps_annealed=steps_annealed,
                                        steps_min=steps_min, Tmax=Tmax, Tmin=Tmin, denoise=denoise, sigma0=sigma0,
                                        record_middle_states=record_middle_states, n_middle_states=n_middle_states)
        for i in range(n):
            fname = os.path.join(self.evaluator_root, "deterministic_interpolate_linear", "%d_%d.png" % (it, i))
            sample = sample_from_dataset(self.train_dataset).to(self.device)
            feature = self.q.expect(sample.unsqueeze(dim=0))[0]
            h = ratio * linear_interpolate(feature, -feature, steps=n_interpolation).to(self.device)
            if record_middle_states:
                samples = sampler.csample_v(h, share_random=True).middle_states
                samples = list(map(lambda x: self.unpreprocess_fn(x[0]), samples))
                samples = torch.stack(samples, dim=0)
                samples = samples.transpose(0, 1).flatten(0, 1)
                grid = make_grid(samples, n_middle_states)
                save_image(grid, fname)
            else:
                samples = sampler.csample_v(h, share_random=True).samples[0]
                samples = self.unpreprocess_fn(samples)
                grid = make_grid(samples, n_interpolation)
                save_image(grid, fname)

    def deterministic_interpolate_impainting_linear(self, it, n=10, sigma=0.02, steps_max=500, steps_annealed=2000, steps_min=200,
                                                    Tmax=100, Tmin=1, denoise=True, sigma0=0.1, ratio=3.,
                                                    mask_type="half", record_middle_states=True,
                                                    n_middle_states=10, n_interpolation=10):
        root = os.path.join(self.evaluator_root, "deterministic_interpolate_impainting_linear")
        os.makedirs(root, exist_ok=True)
        sampler = GeometricAnnealedSGLD(self.lvm, sigma=sigma, steps_max=steps_max, steps_annealed=steps_annealed,
                                        steps_min=steps_min, Tmax=Tmax, Tmin=Tmin, denoise=denoise, sigma0=sigma0,
                                        record_middle_states=record_middle_states, n_middle_states=n_middle_states)
        width = sample_from_dataset(self.train_dataset).shape[-1]
        mnist = Mnist("workspace/datasets/mnist/", width).get_train_data()
        for i in range(n):
            if mask_type == "mnist":
                keep = (sample_from_dataset(mnist).to(self.device) > 0.5).float()
            elif mask_type == "half":
                keep = torch.ones(width, width, dtype=torch.float32, device=self.device)
                keep[:, : width // 2] = 0.
            else:
                raise ValueError
            if random.randint(0, 1):
                keep = 1. - keep
            fname = os.path.join(root, "%d_%d.png" % (it, i))
            sample = sample_from_dataset(self.train_dataset).to(self.device)
            feature = self.q.expect(sample.unsqueeze(dim=0))[0]
            h = ratio * linear_interpolate(feature, -feature, steps=n_interpolation).to(self.device)
            save_image(sample, os.path.join(root, "raw_%d_%d.png" % (it, i)))
            noise = 0.5 + torch.randn_like(sample)
            noise_sample = sample * keep + noise * (1. - keep)
            save_image(noise_sample, os.path.join(root, "noise_%d_%d.png" % (it, i)))
            save_image(sample * keep, os.path.join(root, "mask_%d_%d.png" % (it, i)))
            if record_middle_states:
                noise_sample = func.duplicate(noise_sample, n_interpolation)
                samples = sampler.cimpainting_v(noise_sample, keep, h, share_random=True).middle_states
                samples = list(map(lambda x: self.unpreprocess_fn(x[0]), samples))
                samples = torch.stack(samples, dim=0)
                samples = samples.transpose(0, 1).flatten(0, 1)
                grid = make_grid(samples, n_middle_states)
                save_image(grid, fname)
            else:
                samples = sampler.cimpainting_v(noise_sample, keep, h, share_random=True).samples[0]
                samples = self.unpreprocess_fn(samples)
                grid = make_grid(samples, n_interpolation)
                save_image(grid, fname)
