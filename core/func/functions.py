
__all__ = ["sos", "inner_product", "duplicate", "unsqueeze_like", "logsumexp", "log_normal",
           "binary_cross_entropy_with_logits", "log_bernoulli"]


import numpy as np
import torch.nn.functional as F
import torch


def sos(a, start_dim=1):  # sum of square
    return a.pow(2).flatten(start_dim=start_dim).sum(dim=-1)


def inner_product(a, b, start_dim=1):
    return (a * b).flatten(start_dim=start_dim).sum(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


def unsqueeze_like(tensor, template, start="left"):
    if start == "left":
        tensor_dim = tensor.dim()
        template_dim = template.dim()
        assert tensor.shape == template.shape[:tensor_dim]
        return tensor.view(*tensor.shape, *([1] * (template_dim - tensor_dim)))
    elif start == "right":
        tensor_dim = tensor.dim()
        template_dim = template.dim()
        assert tensor.shape == template.shape[-tensor_dim:]
        return tensor.view(*([1] * (template_dim - tensor_dim)), *tensor.shape)
    else:
        raise ValueError


def logsumexp(tensor, dim, keepdim=False):
    # the logsumexp of pytorch is not stable!
    tensor_max, _ = tensor.max(dim=dim, keepdim=True)
    ret = (tensor - tensor_max).exp().sum(dim=dim, keepdim=True).log() + tensor_max
    if not keepdim:
        ret.squeeze_(dim=dim)
    return ret


def log_normal(inputs, mean, log_std, n_data_dim):
    r"""
    Args:
        inputs: a pytorch tensor
        mean: can be a scalar
        log_std: can be a scalar
        n_data_dim: the last n_data_dim corresponds to data
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(log_std, torch.Tensor):
        log_std = torch.tensor(log_std)
    m = inputs.shape[-n_data_dim:].numel()
    ne = -0.5 * sos((inputs - mean) * (-log_std).exp(), -n_data_dim)
    if log_std.dim() == 0:
        log_partition = 0.5 * m * np.log(2 * np.pi) + m * log_std
    else:
        log_partition = 0.5 * m * np.log(2 * np.pi) + log_std.flatten(-n_data_dim).sum(dim=-1)
    return ne - log_partition


def binary_cross_entropy_with_logits(logits, inputs):
    r""" -inputs * log (sigmoid(logits)) - (1 - inputs) * log (1 - sigmoid(logits)) element wise
        with automatically expand dimensions
    """
    if inputs.dim() < logits.dim():
        inputs = inputs.expand_as(logits)
    else:
        logits = logits.expand_as(inputs)
    return F.binary_cross_entropy_with_logits(logits, inputs, reduction="none")


def log_bernoulli(inputs, logits, n_data_dim):
    return -binary_cross_entropy_with_logits(logits, inputs).flatten(-n_data_dim).sum(dim=-1)
