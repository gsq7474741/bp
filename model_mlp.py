from collections import OrderedDict

import torch
import torch.nn.functional as F
# import nni.retiarii.nn.pytorch as nn
import torch.nn as nn
from nni.retiarii import model_wrapper


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, hidden_dim=None, dropout=0.2, *args, **kwargs):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = self.input_dim * 2 if hidden_dim is None else hidden_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.layers[i](x))
            x = self.dropout(x)
        x = self.out_layer(x)
        x = F.softmax(x, dim=1)
        return x
