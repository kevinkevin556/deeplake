import torch
from torch import nn

from .mmd import MMD


def get_mu(cond1, cond2):
    return cond1.reshape(-1, 1) * cond2.reshape(1, -1) * 1.0


class DCC(nn.Module):
    """class-aware domain discrepancy"""

    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma

    def compute_dcc(self, dist, mu, gamma):
        rbf_kernel_dist = {}
        rbf_kernel_dist["ss"] = torch.exp(-gamma * dist["ss"])
        rbf_kernel_dist["tt"] = torch.exp(-gamma * dist["tt"])
        rbf_kernel_dist["st"] = torch.exp(-gamma * dist["st"])
        e1 = torch.sum(mu["ss"] * rbf_kernel_dist["ss"]) / mu["ss"].sum()
        e2 = torch.sum(mu["tt"] * rbf_kernel_dist["tt"]) / mu["tt"].sum()
        e3 = torch.sum(mu["st"] * rbf_kernel_dist["st"]) / mu["st"].sum()
        return e1 + e2 - 2 * e3

    def forward(self, c1, c2, tensor1=None, tensor2=None, label1=None, label2=None, dist=None):
        if not dist:
            dist = {}
            dist["ss"] = self.compute_pairwise_squared_dist(tensor1, tensor1)
            dist["tt"] = self.compute_pairwise_squared_dist(tensor2, tensor2)
            dist["st"] = self.compute_pairwise_squared_dist(tensor1, tensor2)

        mu = {}
        mu["ss"] = get_mu(label1 == c1, label1 == c1).to(dist["ss"].device)
        mu["tt"] = get_mu(label2 == c2, label2 == c2).to(dist["tt"].device)
        mu["st"] = get_mu(label1 == c1, label2 == c2).to(dist["st"].device)

        gamma = self.estimate_rbf_gamma(dist) if self.gamma is None else self.gamma
        dcc = self.compute_dcc(dist, mu, gamma)
        return dcc


class CDD(MMD):
    """Contrastive Domain Discrepancy"""

    def forward(self, source, target, source_label, target_label, num_classes):
        dist = {}
        dist["ss"] = self.compute_pairwise_squared_dist(source, source)
        dist["tt"] = self.compute_pairwise_squared_dist(target, target)
        dist["st"] = self.compute_pairwise_squared_dist(source, target)

        cdd = 0
        M = num_classes
        dcc = DCC()

        # intra
        for c in range(M):
            cdd += dcc(c1=c, c2=c, label1=source_label, label2=target_label, dist=dist) / M
        # inter
        for c1, c2 in zip(range(M), range(M)):
            if c1 != c2:
                cdd -= dcc(c1, c2, label1=source_label, label2=target_label, dist=dist) / (M * (M - 1))
        return cdd
