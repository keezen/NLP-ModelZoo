# python: 3.5
# encoding: utf-8
# author: zhenke

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss function.

    Inputs
    ------
    logit: torch.float, size [batch, seq_len, num_classes]
    target: torch.long, size [batch, seq_len]
    mask: torch.long, size [batch, seq_len]

    Outputs
    -------
    loss: torch.scalar
    """

    def __init__(self, pos_weight=1.0, reduce=True):
        super(CrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        self.reduce = reduce

    def forward(self, logit, target, mask=None):
        batch_size = logit.size(0)
        num_classes = logit.size(-1)

        loss_vec = self.loss_fn(  # [b*sl]
            logit.contiguous().view(-1, num_classes),
            target.contiguous().view(-1))
        loss_mat = loss_vec.view_as(target)  # [b,sl]

        if mask is not None:  # [b,sl]
            loss_mat = loss_mat * mask.float()

        ones = torch.ones_like(target).to(logit)
        weight = torch.where(
            target > 0, ones * self.pos_weight, ones).to(logit)  # [b,sl]
        loss_mat = loss_mat * weight

        if self.reduce:
            loss = torch.mean(
                torch.sum(loss_mat.view(batch_size, -1), dim=-1))  # scalar
        else:
            loss = loss_mat

        return loss


class BCELoss(nn.Module):
    """Loss function for binary classification.

    Inputs
    ------
    logits: torch.float, size [batch, seq_len]
    target: torch.long, size [batch, seq_len]
    mask: torch.long, size [batch, seq_len]

    Outputs
    -------
    loss: torch.scalar. Loss for trigger detection.
    """

    def __init__(self, pos_weight=1.0, reduce=True):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False)
        self.pos_weight = pos_weight
        self.reduce = reduce

    def forward(self, logits, target, mask=None):
        batch_size = logits.size(0)

        # target and weight
        weight = torch.ones_like(target).float()  # [b,ntc]
        weight = torch.where(target > 0, weight * self.pos_weight, weight)

        # loss
        loss_mat = self.loss_fn(logits, target.float())  # [b,ntc]
        loss_mat = loss_mat * weight
        if mask is not None:
            loss_mat = loss_mat * mask.float()

        if self.reduce:
            loss = torch.mean(
                torch.sum(loss_mat.view(batch_size, -1), dim=-1))
        else:
            loss = loss_mat

        return loss
