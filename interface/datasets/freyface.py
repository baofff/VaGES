from .dataset_factory import DatasetFactory
from .utils import *
import os
import urllib.request as request
import scipy.io as sio
import numpy as np
import torch
from core.utils import global_device


class FreyFace(DatasetFactory):
    r""" FreyFace dataset

    Information of the raw dataset:
         train: 1,400
         val:   300
         test:  265
         shape: 28 * 20
         train mean: 0.6049
         train biased std: 0.1763
    """

    def __init__(self, data_path, binarized=False, gauss_noise=False, noise_std=0.01, padding=False,
                 normalize=None, flattened=True, **kwargs):
        super().__init__()
        self.binarized = binarized
        self.gauss_noise = gauss_noise
        self.noise_std = noise_std
        self.padding = padding
        self.pad = [6, 6, 2, 2]
        self.normalize = normalize
        self.flattened = flattened
        self.device = global_device()

        if self.binarized:
            assert not self.gauss_noise
            assert self.normalize is None
        assert self.normalize == "standardize" or self.normalize == "subtract_mean" or self.normalize is None

        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.fname = os.path.join(self.data_path, "frey_rawface.mat")
        if not os.path.exists(self.fname):
            request.urlretrieve("https://cs.nyu.edu/~roweis/data/frey_rawface.mat", self.fname)

        data = sio.loadmat(self.fname)['ff'].transpose().astype(np.float32) / 255.
        data = torch.tensor(data, device=self.device).view(-1, 1, 28, 20)
        train_data, val_data, test_data = data[: 1400], data[1400: 1700], data[1700:]

        # Get datasets
        self.train = QuickDataset(train_data)
        self.val = QuickDataset(val_data)
        self.test = QuickDataset(test_data)
        self.uniform = QuickDataset(torch.rand_like(train_data))

        # Calculate the train mean and std
        if self.normalize == "standardize":
            standardize_mode = kwargs.get("standardize_mode", default="pixel")
            if standardize_mode == "pixel":
                self.train_mean = 0.6049
                self.train_std = 0.1763
            elif standardize_mode == "vector":
                self.train_mean = train_data.mean(dim=0)
                self.train_std = train_data.std(dim=0, unbiased=False) + 1e-3
            else:
                raise NotImplementedError

    def distribution_transform(self, dataset):
        if self.binarized:
            dataset = BinarizedDataset(dataset)
        if self.gauss_noise:
            dataset = GaussNoiseDataset(dataset, std=self.noise_std)
        return dataset

    def affine_transform(self, dataset):
        if self.padding:
            dataset = PaddedDataset(dataset, pad=self.pad)
        if self.normalize == "standardize":
            dataset = StandardizedDataset(dataset, mean=self.train_mean, std=self.train_std)
        elif self.normalize == "subtract_mean":
            dataset = TranslatedDataset(dataset, delta=-self.train_mean)
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def preprocess(self, v):
        if self.padding:
            v = F.pad(v, self.pad)
        if self.normalize == "standardize":
            v = (v - self.train_mean) / self.train_std
        elif self.normalize == "subtract_mean":
            v = v - self.train_mean
        if self.flattened:
            v = v.flatten(1)
        return v

    def unpreprocess(self, v):
        if self.padding:
            v = v.view(len(v), 1, 32, 32)
        else:
            v = v.view(len(v), 1, 28, 20)
        if self.normalize == "standardize":
            v *= self.train_std
            v += self.train_mean
        if self.normalize == "subtract_mean":
            v += self.train_mean
        if self.padding:
            v = v[..., 2:-2, 6:-6]
        v.clamp_(0., 1.)
        return v
