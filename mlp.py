# -*- coding: utf-8 -*-

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0.0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
