from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, data_path, binarized=False, gauss_noise=False, noise_std=0.01, flattened=False, width=32):
        super(CelebA, self).__init__()
        self.binarized = binarized
        self.gauss_noise = gauss_noise
        self.noise_std = noise_std
        self.flattened = flattened
        self.width = width

        if self.binarized:
            assert not self.gauss_noise

        self.data_path = data_path

        _transform = [transforms.CenterCrop(140), transforms.Resize(self.width),
                      transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
        if self.binarized:
            _transform.append(Binarize())
        if self.gauss_noise:
            _transform.append(AddGaussNoise(self.noise_std))
        im_transform = transforms.Compose(_transform)
        self.train = datasets.CelebA(self.data_path, split="train", target_type=[],
                                     transform=im_transform, download=True)
        self.val = datasets.CelebA(self.data_path, split="valid", target_type=[],
                                   transform=im_transform, download=True)
        self.test = datasets.CelebA(self.data_path, split="test", target_type=[],
                                    transform=im_transform, download=True)

        self.train = UnlabeledDataset(self.train)
        self.test = UnlabeledDataset(self.test)
        self.val = UnlabeledDataset(self.val)

    def affine_transform(self, dataset):
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def preprocess(self, v):
        if self.flattened:
            v = v.flatten(1)
        return v

    def unpreprocess(self, v):
        v = v.view(len(v), 3, self.width, self.width)
        v.clamp_(0., 1.)
        return v
