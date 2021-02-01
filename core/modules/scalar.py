
__all__ = ["LinearAFSquare", "MLPAFSquare", "Quadratic", "AFQuadratic", "LinearAFQuadratic"]


import torch.nn as nn
from typing import List
from .mlp import MLP


class LinearAFSquare(nn.Module):
    def __init__(self, in_features, features, af=nn.ELU()):
        super(LinearAFSquare, self).__init__()
        self.linear = nn.Linear(in_features, features)
        self.af = af

    def forward(self, inputs):
        return self.af(self.linear(inputs)).pow(2).sum(-1)


class MLPAFSquare(nn.Module):
    def __init__(self, n_features_lst: List[int], af=nn.ELU()):
        super().__init__()
        self.mlp = MLP(n_features_lst, af)
        self.af = af

    def forward(self, inputs):
        return self.af(self.mlp(inputs)).pow(2).sum(-1)


class Quadratic(nn.Module):
    def __init__(self, in_features):
        super(Quadratic, self).__init__()
        self.linear1 = nn.Linear(in_features, 1)
        self.linear2 = nn.Linear(in_features, 1)
        self.linear3 = nn.Linear(in_features, 1)

    def forward(self, inputs):
        return (self.linear1(inputs) * self.linear2(inputs) + self.linear3(inputs.pow(2))).squeeze(dim=-1)


class AFQuadratic(nn.Module):
    def __init__(self, in_features, af=nn.ELU()):
        super(AFQuadratic, self).__init__()
        self.af = af
        self.quadratic = Quadratic(in_features)

    def forward(self, inputs):
        return self.quadratic(self.af(inputs))


class LinearAFQuadratic(nn.Module):
    def __init__(self, in_features, features):
        super(LinearAFQuadratic, self).__init__()
        self.linear = nn.Linear(in_features, features)
        self.af_quadratic = AFQuadratic(features)

    def forward(self, inputs):
        return self.af_quadratic(self.linear(inputs))
