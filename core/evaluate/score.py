
__all__ = ["score_on_dataset", "reconstruct_error", "negative_log_likelihood_score_4_grbm"]

import torch
import core.utils.managers as managers
import core.func as func
from torch.utils.data import DataLoader, Dataset
from core.inference.ll import AIS4GRBM


def score_on_dataset(dataset: Dataset, score_fn, batch_size):
    r"""
    Args:
        dataset: an instance of Dataset
        score_fn: a batch of data -> a batch of scalars
        batch_size: the batch size
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    total_score = 0.
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for v in dataloader:
        v = v.to(device)
        score = score_fn(v)
        total_score += score.sum().detach()
    mean_score = total_score / len(dataset)
    return mean_score


################################################################################
# Score on tensor
################################################################################

def reconstruct_error(models: managers.ModelsManager, v: torch.Tensor):
    r""" The reconstruct error
    Args:
        models: an instance of ModelsManager
        v: the tested data
    """
    lvm = models.lvm
    h = lvm.csample_h(v, n_particles=1).squeeze(dim=0)
    v_reconstruct = lvm.csample_v(h)
    return func.sos(v - v_reconstruct)


################################################################################
# Score on dataset
################################################################################

def negative_log_likelihood_score_4_grbm(models: managers.ModelsManager, dataset: Dataset, batch_size: int):
    ais = AIS4GRBM(models.lvm)
    ais.update_log_partition()
    ll_score = score_on_dataset(dataset, score_fn=ais.estimate_ll, batch_size=batch_size)
    return -ll_score
