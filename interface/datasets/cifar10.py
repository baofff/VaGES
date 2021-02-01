from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *


class Cifar10(DatasetFactory):
    r""" Cifar10 dataset

    Information of the raw dataset:
         train: 40,000
         val:   10,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, data_path, binarized=False, gauss_noise=False, noise_std=0.01, flattened=False):
        super(Cifar10, self).__init__()
        self.data_path = data_path
        self.binarized = binarized
        self.gauss_noise = gauss_noise
        self.noise_std = noise_std
        self.flattened = flattened

        if self.binarized:
            assert not self.gauss_noise

        _transform = [transforms.ToTensor()]
        if self.binarized:
            _transform.append(Binarize())
        if self.gauss_noise:
            _transform.append(AddGaussNoise(self.noise_std))
        im_transform = transforms.Compose(_transform)
        self.train_val = datasets.CIFAR10(self.data_path, train=True, transform=im_transform, download=True)
        self.train = Subset(self.train_val, list(range(40000)))
        self.val = Subset(self.train_val, list(range(40000, 50000)))
        self.test = datasets.CIFAR10(self.data_path, train=False, transform=im_transform, download=True)

    def affine_transform(self, dataset):
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def preprocess(self, v):
        if self.flattened:
            v = v.flatten(1)
        return v

    def unpreprocess(self, v):
        v = v.view(len(v), 3, 32, 32)
        v.clamp_(0., 1.)
        return v
