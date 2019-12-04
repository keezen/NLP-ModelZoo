# python: 3.6
# encoding: utf-8
# pytorch: 1.0.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseLayerLinear(nn.Module):
    """Layer to fuse uni-gram feature and bi-gram feature,
    with linear projection.
    """

    def __init__(self, uni_size, bi_size, out_size, dropout=0):
        super().__init__()
        self.linear = nn.Linear(uni_size + bi_size, out_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_size)

    def forward(self, uni, bi):
        """
        :param uni: torch.float, [b,l,h]
        :param bi: torch.float, [b,l,h]
        :return out: torch.float, [b,l,h]
        """
        out = self.linear(torch.cat([uni, bi], dim=-1))  # [b,l,h]
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out


class FuseLayerGate(nn.Module):
    """Layer to fuse two representations with fusion gate.
    """

    def __init__(self, in_size1, in_size2, out_size, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(in_size1, out_size)
        self.linear2 = nn.Linear(in_size2, out_size)
        self.gate = nn.Linear(in_size1 + in_size2, out_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_size)

    def forward(self, in1, in2):
        """
        :param in1: torch.float, [b,l,h]
        :param in2: torch.float, [b,l,h]
        :return out: torch.float, [b,l,h]
        """
        gate = self.gate(torch.cat([in1, in2], dim=-1))  # [b,l,h]
        gate = torch.sigmoid(gate)

        out1 = self.linear1(in1)  # [b,l,h]
        out1 = torch.tanh(out1)
        out2 = self.linear2(in2)  # [b,l,h]
        out2 = torch.tanh(out2)

        out = gate * out1 + (1.0 - gate) * out2  # [b,l,h]

        out = self.dropout(out)
        out = self.layer_norm(out)
    
        return out
