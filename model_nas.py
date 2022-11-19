from collections import OrderedDict

import torch
import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
# import torch.nn as nn
from nni.retiarii import model_wrapper


@model_wrapper  # this decorator should be put on the out most
class NasNet(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        # self.op_choice_list = OrderedDict([
        #     ("linear", nn.Conv2d(3, 16, 128)),
        #     ("dropout", nn.Conv2d(5, 16, 128)),
        #     ("activation", nn.Conv2d(7, 16, 128))
        # ])
        self.hidden_layer_num = 10
        self.layers = nn.ModuleList()
        # self.blocks = nn.Repeat(lambda index: nn.LayerChoice(
        #     [nn.Linear(self.layers[index - 1].out_features, nn.ValueChoice([i for i in range(1, 2000)])),
        #      nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75])),
        #      nn.ReLU()],
        #     label=f'layer{index}'), nn.ValueChoice([i for i in range(3, 20)]))

        for i in range(self.hidden_layer_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, nn.ValueChoice([i for i in range(1, 2000)])))
            # elif i == self.hidden_layer_num - 1:
            #     self.layers.append(nn.Linear(nn.ValueChoice([i for i in range(output_dim, 2000)]), output_dim))
            else:
                # self.layers.append(nn.LayerChoice(
                #     [nn.Linear(self.layers[i - 1].out_features, nn.ValueChoice([i for i in range(1, 2000)])),
                #      nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75])),
                #      nn.ReLU()
                #      ]
                # ))
                self.layers.append(
                    nn.Linear(self.layers[i - 1].out_features, nn.ValueChoice([i for i in range(1, 2000)])))
        self.layers.append(nn.Linear(self.layers[-1].out_features, output_dim))
        # self.out_layer = nn.Linear(self.layers[-1].out_features, self.output_dim)
        # self.layers.append(nn.AutoActivation())
        self.dropout = nn.Dropout(nn.ValueChoice([0.25, 0.5, 0.75]))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        # output = F.softmax(x, dim=1)
        return x


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
