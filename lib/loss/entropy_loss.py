import numpy as np
import torch
from torch.nn.modules.loss import _Loss


def get_entropy_map(v, eps=0, keepdim=True, logits=True):
    """
    Entropy map for probabilistic prediction vectors
    input: batch_size x channels x h x w
    output: batch_size x 1 x h x w
    """
    _, c, _, _ = v.size()
    if logits:
        p = torch.softmax(v, dim=1)
        log_p = torch.log_softmax(v, dim=1) / np.log(2)
    else:
        p = v
        log_p = torch.log2(v + eps)
    ent_map = torch.sum(torch.mul(p, log_p), dim=1, keepdim=keepdim) / np.log2(c)
    return ent_map


class EntropyLoss(_Loss):
    def __init__(self, eps=1e-30, logits=True):
        super().__init__()
        self.eps = eps
        self.logits = logits

    def forward(self, x):
        return torch.mean(get_entropy_map(x, eps=self.eps, logits=self.logits))
