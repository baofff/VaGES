from .dataset_factory import DatasetFactory
from .utils import *
import torch
from sklearn.datasets import make_blobs, make_swiss_roll, make_moons
from interface.utils.plot import plot_scatter
import numpy as np


def my_make_blobs(n_samples, centers, std):
    x, _ = make_blobs(n_samples, centers=centers, cluster_std=std)
    return torch.tensor(x).float()


def make_mark_points(n_cols):
    dx = 1. / n_cols
    mark_points = []
    for i in range(n_cols):
        for j in range(n_cols):
            if (i + j) % 2 == 0:
                mark_points.append((i * dx, j * dx))
    return torch.tensor(mark_points)


def make_checkerboard(n_samples, n_cols):
    dx = 1. / n_cols
    mark_points = make_mark_points(n_cols)
    data = torch.rand(n_samples, 2) * dx
    positions = torch.randint(0, len(mark_points), (n_samples,))
    for i in range(n_samples):
        data[i] = data[i] + mark_points[positions[i]]
    return data


def my_make_swiss_roll(n_samples):
    x, _ = make_swiss_roll(n_samples)
    return torch.tensor(x[:, [0, 2]]).float()


def my_make_moons(n_samples):
    x, _ = make_moons(n_samples)
    return torch.tensor(x).float()


def make_circle(n_samples):
    thetas = torch.linspace(0, 2 * np.pi, n_samples + 1)[:-1]
    x = thetas.cos()
    y = thetas.sin()
    return torch.stack([x, y], dim=1)


class Toy(DatasetFactory):
    def __init__(self, data_type, **kwargs):
        super().__init__()
        self.data_type = data_type
        if self.data_type == "diamond_blobs":
            centers = [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]
            std = kwargs.get("std", 0.1)
            fn = lambda n: my_make_blobs(n, centers, std)
        elif self.data_type == "triangle_blobs":
            centers = [[0., 1.], [3 ** 0.5 / 2., -0.5], [-3 ** 0.5 / 2., -0.5]]
            std = kwargs.get("std", 0.1)
            fn = lambda n: my_make_blobs(n, centers, std)
        elif self.data_type == "line_blobs":
            centers = [[-1., -1.], [0., 0.], [1., 1.]]
            std = kwargs.get("std", 0.1)
            fn = lambda n: my_make_blobs(n, centers, std)
        elif self.data_type == "two_blobs":
            centers = [[-1., -1.], [1., 1.]]
            fn = lambda n: my_make_blobs(n, centers, [0.1, 0.2])
        elif self.data_type == "checkerboard":
            n_cols = kwargs.get("n_cols", 2)
            fn = lambda n: make_checkerboard(n, n_cols)
        elif self.data_type == "swiss_roll":
            fn = my_make_swiss_roll
        elif self.data_type == "moons":
            fn = my_make_moons
        elif self.data_type == "circle":
            fn = make_circle
        else:
            raise NotImplementedError
        self.train = QuickDataset(fn(50000))
        self.val = QuickDataset(fn(10000))
        self.test = QuickDataset(fn(10000))

    def plot_dataset(self, fname):
        assert self.train[0].dim() == 1 and self.train[0].size(0) == 2
        plot_scatter(self.train[:], fname)
