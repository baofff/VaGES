
__all__ = ["Const"]


import torch.nn as nn
import torch


class Const(nn.Module):
    def __init__(self):
        r""" A constant module, used in NCE and VNCE
        """
        super().__init__()
        self.c = nn.Parameter(torch.scalar_tensor(0.))

    def forward(self):
        return self.c
