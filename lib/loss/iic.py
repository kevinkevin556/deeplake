# https://arxiv.org/pdf/1807.06653
# Modified from: https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py

import sys

import torch
from torch.nn.modules.loss import _Loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
    return p_i_j


class IIDLoss(_Loss):
    def __init__(self, lamb=1.0, eps=sys.float_info.epsilon):
        super().__init__()
        self.lamb = lamb
        self.eps = eps

    def forward(self, x_out, x_tf_out, num_classes=None):
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert p_i_j.size() == (k, k)

        k = num_classes if num_classes is not None else x_out.shape[1]
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[(p_i_j < self.eps).data] = self.eps
        p_j[(p_j < self.eps).data] = self.eps
        p_i[(p_i < self.eps).data] = self.eps

        loss = -p_i_j * (torch.log(p_i_j) - self.lamb * torch.log(p_j) - self.lamb * torch.log(p_i))
        loss = loss.sum()
        return loss


class FeatureSpaceIIDLoss(_Loss):
    def __init__(self, lamb=1.0, eps=sys.float_info.epsilon, num_classes=None, softmax=True):
        super().__init__()
        self.lamb = lamb
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, feature):
        pass
