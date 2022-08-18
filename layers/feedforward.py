import torch
from torch import nn
import torch.nn.functional as F

from typing import MutableSequence
from .utils import get_activation, get_normalization

class MLP(nn.Module):
    def __init__(self, 
            in_size, 
            hidden_sizes, 
            dropout=0,
            normalization=False,
            activation='relu', 
            last_dropout=None,
            last_normalization=None,
            last_activation=None):
        super(MLP, self).__init__()
        
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        if last_dropout is None:
            last_dropout = dropout
        if last_normalization is None:
            last_normalization = normalization
        if last_activation is None:
            last_activation = activation
        
        assert isinstance(hidden_sizes, MutableSequence)     
        # assert activation in ['sigmoid', 'tanh', 'relu']
        # assert last_activation in ['identity', 'sigmoid', 'tanh', 'relu']
        
        layers = []
        num_layers = len(hidden_sizes)

        for l, hid_size in enumerate(hidden_sizes):
            # add linear layer
            layers.append(nn.Linear(in_size, hid_size))

            # add normalization
            norm = normalization if l < num_layers - 1 else last_normalization
            if norm:
                layers.append(get_normalization(norm, hid_size))

            # add activation
            act = activation if l < num_layers - 1 else last_activation
            layers.append(get_activation(act))

            # add dropout
            p = dropout if l < num_layers - 1 else last_dropout
            if p > 0:
                layers.append(nn.Dropout(p))

            in_size = hid_size

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MultiColumnMLP(nn.Module):
    def __init__(self, 
            in_size, 
            hidden_sizes, 
            num_columns,
            dropout=0,
            activation='relu', 
            last_dropout=None,
            last_activation=None):
        super(MultiColumnMLP, self).__init__()

        columns = []
        for c in range(num_columns):
            mlp = MLP(
                in_size,
                hidden_sizes,
                dropout=dropout,
                activation=activation,
                last_dropout=last_dropout,
                last_activation=last_activation
            )
            columns.append(mlp)
        
        self.columns = nn.ModuleList(columns)

    def forward(self, x):
        y = []
        for column in self.columns:
            y.append(column(x))
        return torch.cat(y, dim=1)

# class ParameterizedStimLinear(nn.Module):
#     def __init__(self, in_features, out_features, activation='relu'):
#         super(ParameterizedStimLinear, self).__init__()
#         self.activation = None if activation is None else getattr(F, activation)
#         self.linear = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         y = self.linear(x)

#         if self.activation is not None:
#             y = self.activation(y)

#         return y
