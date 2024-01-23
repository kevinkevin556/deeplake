from typing import Literal

import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.nn import NLLLoss
from torch.nn.modules.loss import _Loss


def channelwise_matmul(input, other):
    if len(input.shape) == 4:
        perm_index = (0, 2, 3, 1)
        inv_perm_index = (0, 3, 1, 2)
    elif len(input.shape) == 5:
        perm_index = (0, 2, 3, 4, 1)
        inv_perm_index = (0, 4, 1, 2, 3)
    else:
        raise ValueError("Invalid input dimension.")
    input_reshaped = input.permute(*perm_index)
    output = torch.matmul(input_reshaped, other).permute(*inv_perm_index)
    return output


class TargetAdaptativeLoss(_Loss):
    def __init__(self, num_classes: int, background_classes: list, device: Literal["cuda", "cpu"] = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        if isinstance(background_classes, (Tensor, ndarray)):
            self.background = background_classes.tolist()
        else:
            self.background = background_classes
        assert isinstance(self.background, (list, tuple))
        self.foreground = list(set(range(1, self.num_classes)) - set(self.background))

        # probability mergeing matrix
        self.prob_merge_mat = torch.ones((num_classes, 1), requires_grad=False)
        self.prob_merge_mat[self.foreground, 0] = 0
        self.prob_merge_mat = self.prob_merge_mat.to(device)

        # Weighting matrix
        self.mat_a = torch.eye(num_classes, num_classes)
        self.mat_a[0, 0] = 0
        self.mat_a[self.background, self.background] = 0
        self.mat_a = self.mat_a.to(device)

        self.mat_b = torch.zeros(1, num_classes)
        for i in [0] + self.background:
            self.mat_b[0, i] = 1
        self.mat_b = self.mat_b.to(device)

        # nll
        # weight = torch.Tensor([0 if i in self.background else 1 for i in range(num_classes)]).to(device)
        self.nll = NLLLoss()

    def forward(self, logits, target):
        logits_ch, target_ch = logits.shape[1], target.shape[1]
        if logits_ch != target_ch and target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()

        log_prob = F.log_softmax(logits, dim=1)
        fg_log_prob = channelwise_matmul(log_prob, self.mat_a).contiguous()

        prob = torch.exp(log_prob)
        bg_log_prob_value = torch.log(channelwise_matmul(prob, self.prob_merge_mat)).contiguous()
        bg_log_prob = channelwise_matmul(bg_log_prob_value, self.mat_b).contiguous()

        log_prob_m = fg_log_prob + bg_log_prob
        loss = self.nll(log_prob_m, target)
        return loss

    def __repr__(self):
        return (
            "TargetAdaptativeLoss(\n"
            f"  (num_classes): {self.num_classes}\n"
            f"  (foreground): {self.foreground}\n"
            f"  (background): {self.background}\n"
            f"  (nll): {self.nll}\n"
            ")"
        )
