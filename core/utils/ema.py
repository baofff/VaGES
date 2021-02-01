import torch.nn as nn


def ema(model_dest: nn.Module, model_src: nn.Module, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(beta).add_((1 - beta) * p_src.data)
