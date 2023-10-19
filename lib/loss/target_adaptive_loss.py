import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from numpy import ndarray
from torch import Tensor
from torch.nn import NLLLoss
from torch.nn.modules.loss import _Loss


class TargetAdaptiveLoss(_Loss):
    def __init__(self, num_classes, foreground, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.foreground = foreground.tolist() if isinstance(foreground, (Tensor, ndarray)) else foreground
        self.background = list(set(range(1, self.num_classes)) - set(self.foreground))

        # probability mergeing matrix
        self.prob_merge_mat = torch.ones((num_classes, 1), requires_grad=False)
        self.prob_merge_mat[self.foreground, 0] = 0
        self.prob_merge_mat = self.prob_merge_mat.to(device)

        # Weighting matrix
        self.mat_a = torch.eye(num_classes, num_classes)
        self.mat_a[0, 0] = 0
        self.mat_a[self.background, self.background] = 0
        self.mat_a = self.mat_a.to(device)

        self.mat_b = torch.zeros(num_classes, num_classes)
        for i in [0] + self.background:
            self.mat_b[:, i] = torch.Tensor([1 if j not in self.foreground else 0 for j in range(num_classes)])
        self.mat_b = self.mat_b.to(device)

        # nll
        # weight = torch.Tensor([0 if i in self.background else 1 for i in range(num_classes)]).to(device)
        self.nll = NLLLoss()

    def forward(self, logits, target):
        n_pred_ch, n_target_ch = logits.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()

        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        ch_multiply = "n a ..., a b -> n b ..."
        bg_log_prob = torch.log(einsum(prob, self.prob_merge_mat, ch_multiply))
        log_prob_m = einsum(log_prob, self.mat_a, ch_multiply) + einsum(bg_log_prob, self.mat_b, ch_multiply)
        # calculate loss
        loss = self.nll(log_prob_m, target)
        return loss

    def __repr__(self):
        return (
            "TargetAdaptiveLoss(\n"
            f"  (num_classes): {self.num_classes}\n"
            f"  (foreground): {self.foreground}\n"
            f"  (background): {self.background}\n"
            f"  (nll): {self.nll}\n"
            ")"
        )
