import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn.modules.loss import _Loss


class TargetAdaptiveLoss(_Loss):
    def __init__(self, num_class, foreground):
        super().__init__()
        self.criterion = NLLLoss()
        self.foreground = foreground
        self.background = list(set(range(num_class)) - set(foreground))
        # probability mergeing matrix
        self.prob_merge_mat = torch.eye(num_class, len(foreground) + 1)
        self.prob_merge_mat[self.background, 0] = 1

    def forward(self, logits, target):
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_m = einsum(log_prob, self.prob_merge_mat, "n c1 ..., c1 c2 -> n c2 ...")
        log_prob_m = rearrange(log_prob_m, "n c ... -> (n ...) c")
        target = rearrange(target, "n ... -> (n ...)")
        return self.criterion(log_prob_m, target)
