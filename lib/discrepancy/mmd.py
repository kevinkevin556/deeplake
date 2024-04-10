import numpy as np
import torch
from torch import Tensor, nn


def get_tril_values(x, offset=-1):
    return torch.Tensor(x)[torch.tril_indices(*x.shape, offset=offset).unbind()]


class MMD(nn.Module):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma

    def compute_pairwise_squared_dist(self, A: Tensor, B: Tensor) -> Tensor:
        # https://alex.smola.org/posts/4-second-binomial/
        # This is an unstable implementation:
        #     a_norm2 = torch.norm(A, p=2, dim=1) ** 2
        #     b_norm2 = torch.norm(B, p=2, dim=1) ** 2
        #     dist = a_norm2.reshape(-1, 1) + b_norm2.reshape(1, -1) - 2 * A @ B.T
        #     return dist
        return torch.cdist(A[None], B[None])[0]

    def estimate_rbf_gamma(self, dist, default_gamma=1):
        # Calculate parameter gamma using median heuristic
        # https://arxiv.org/pdf/1707.07269.pdf
        dist_ss = get_tril_values(dist["ss"]).flatten()
        dist_tt = get_tril_values(dist["tt"]).flatten()
        dist_st = dist["st"].flatten()
        median_dist = torch.median(torch.cat([dist_ss, dist_tt, dist_st])).item()
        if np.allclose(median_dist, 0):
            gamma = default_gamma
        else:
            gamma = 1 / (0.5 * median_dist) ** 0.5
        return gamma

    def compute_mmd(self, dist, gamma):
        rbf_kernel_dist = {}
        rbf_kernel_dist["ss"] = torch.exp(-gamma * dist["ss"])
        rbf_kernel_dist["tt"] = torch.exp(-gamma * dist["tt"])
        rbf_kernel_dist["st"] = torch.exp(-gamma * dist["st"])
        mmd = (
            torch.mean(rbf_kernel_dist["ss"])
            + torch.mean(rbf_kernel_dist["tt"])
            - 2.0 * torch.mean(rbf_kernel_dist["st"])
        )
        return mmd

    def forward(self, source: Tensor, target: Tensor):
        dist = {}
        dist["ss"] = self.compute_pairwise_squared_dist(source, source)
        dist["tt"] = self.compute_pairwise_squared_dist(target, target)
        dist["st"] = self.compute_pairwise_squared_dist(source, target)
        gamma = self.estimate_rbf_gamma(dist) if self.gamma is None else self.gamma
        mmd = self.compute_mmd(dist, gamma)
        return mmd
