import torch
from torch import nn

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'identity':
        return nn.Identity()
    else:
        raise NotImplementedError

def get_normalization(normalization, normalized_shape):
    if normalization == 'ln':
        return nn.LayerNorm(normalized_shape)
    elif normalization == 'bn1d':
        return nn.BatchNorm1d(normalized_shape)
    elif normalization == 'bn2d':
        return nn.BatchNorm2d(normalized_shape)
    else:
        raise NotImplementedError
