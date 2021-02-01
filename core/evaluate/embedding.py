
__all__ = ["dimension_reduction", "variational_posterior_extract_feature",
           "variational_posterior_2d_embedding", "plot_variational_posterior_embedding"]


import core.utils as utils
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import os
import torch
import matplotlib.pyplot as plt


def dimension_reduction(inputs):
    model = TSNE(n_components=2, random_state=0)
    features = model.fit_transform(inputs)
    return features


def variational_posterior_extract_feature(q, dataset, labelled, batch_size):
    if labelled:
        features, labels = [], []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for v, y in dataloader:
            v = v.to(utils.device_of(q))
            feature = q.expect(v).detach().cpu().numpy()
            features.append(feature)
            labels.append(y)
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels
    else:
        features = []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for v in dataloader:
            v = v.to(utils.device_of(q))
            feature = q.expect(v).detach().cpu().numpy()
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return features


def variational_posterior_2d_embedding(q, dataset, labelled, batch_size):
    if labelled:
        features, labels = variational_posterior_extract_feature(q, dataset, labelled, batch_size)
        embeddings = dimension_reduction(features)
        clusters = {}
        for feature, y in zip(embeddings, labels):
            if int(y) not in clusters:
                clusters[int(y)] = []
            clusters[int(y)].append(feature)
        clusters = {k: np.array(val) for k, val in clusters.items()}
    else:
        features = variational_posterior_extract_feature(q, dataset, labelled, batch_size)
        embeddings = dimension_reduction(features)
        clusters = {0: embeddings}
    return clusters


def plot_variational_posterior_embedding(fname, q, dataset, labelled, batch_size):
    root, name = os.path.split(fname)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "tensor"), exist_ok=True)
    clusters = variational_posterior_2d_embedding(q, dataset, labelled, batch_size)
    torch.save(clusters, os.path.join(root, "tensor", "%s.pth" % name))
    for k, val in clusters.items():
        plt.scatter(val[:, 0], val[:, 1], label="{}".format(k))
    plt.legend()
    plt.savefig(fname)
    plt.close()
