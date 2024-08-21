import torch
import torch.nn.functional as F

from lib.tensor_shape import tensor


def get_prototype_mapping(
    features: tensor["n d"],
    prototypes: tensor["k d"],
    smoothness: float,
    cluster_dist: tensor["k 1"] | None = None,
    niter=None,
    eps=None,
):
    """
    Get the pixel-to-prototype mapping in ProtoSeg (https://arxiv.org/pdf/2203.15102)
    or the code matrix of SwAV (https://arxiv.org/pdf/2006.09882)
    with Sinkhorn-Knopp iteration.

    args:
        features: A (N, D) tensor. N stands for the batch size.
        prototypes: A (K, D) tensor. K is the number of prototypes and D is the dimension of feature space.
        smoothness: the parameter that controls the smoothness of distribution.
            The notation is epsilon in the SwAV paper and kappa in ProtoSeg.
        cluster_dist: A (K, 1) tensor. If None, the cluster distributes uniformly.
        niter: The number of SK iteration.
        eps: The stopping deviation of SK iteration. Either `niter` or `eps` should be specified.

    return:
        Q: A (K, N) tensor, the code matrix / the mapping from instance to prototypes.
            The notation is $L_c$ in ProtoSeg and $Q$ in SwAV.
    """
    n = features.shape[0]
    k = prototypes.shape[0]
    assert (
        features.shape[1] == prototypes.shape[1]
    ), "The number of dimensions of features should be the same as that of prototypes."

    solver = SinkhornKnopp(niter, eps)

    M: tensor[k, n] = prototypes @ features.T
    lbd = 1 / smoothness

    sample_dist: tensor[n, 1] = torch.ones(n, 1) / n
    if cluster_dist is None:
        cluster_dist: tensor[k, 1] = torch.ones(k, 1) / k

    Q: tensor[k, n] = solver.get_P(M, lbd, r=cluster_dist, c=sample_dist)
    return Q


def self_labelling(logits, lbd, cluster_dist=None, ncluster=None, niter=None, eps=None):
    """
    Self-labelling via simultaneous clustering and representation learning

    Solved with Sinkhorn-Knopp iteration.

    args:
        logits: A (N, D) tensor. N stands for the number of samples.
        ldb: the parameter of langrange multiplier lambda
        cluster_dist: A (K, 1) tensor. If None, `ncluster` should be assigned a int.
        ncluster: The number of clusters.
        niter: The number of SK iteration.
        eps: The stopping deviation of SK iteration. Either `niter` or `eps` should be specified.

    return:
        Q: A (K, N) tensor.
    """

    n = logits.shape[0]
    k = cluster_dist.shape[0] if cluster_dist is not None else ncluster
    assert k is not None, "Either cluster_dist or ncluster should be specified. "

    solver = SinkhornKnopp(niter, eps)

    P: tensor[k, n] = F.softmax(logits)

    sample_dist = tensor[n, 1] = torch.ones(n, 1) / n
    if cluster_dist is None:
        cluster_dist: tensor[k, 1] = torch.ones(k, 1) / k

    Q: tensor[k, n] = solver.get_P(-torch.log(P), lbd, r=cluster_dist, c=sample_dist)
    return Q


class SinkhornKnopp:
    def __init__(self, niter=None, eps=None):
        self.u = None
        self.v = None
        self.K = None

        assert (niter is not None) or (eps is not None)
        self.eps = eps
        self.niter = niter

    def get_P(self, M=None, lbd=None, r=None, c=None):
        """
        Obtain the optimal joint probility matrix of entropic OT problem.

        args:
            M: A (n, n) Tensor, cost matrix
            ldb: the parameter of langrange multiplier lambda
            r: A (n, 1) Tensor, which is the vector of source histogram
            C: A (n, m) Tensor, which is the collection of m target histograms

        return:
            out: A (m, n, n) Tensor. If m == 1, then a (n, n) Tensor is returned.
        """
        # Solve the OT problem if inputs are provided
        if (M is not None) and (lbd is not None) and (r is not None) and (c is not None):
            self.solve(M, lbd, r, c)
        elif (self.u is not None) and (self.v is not None) and (self.K is not None):
            pass
        else:
            raise ValueError(
                "The value of u, v, and K should be computed in advance. Otherwise inputs should be provided."
            )

        n, m = self.u.shape

        # Compute P based on Sinkhorn’s theorem:: P = diag(u) K diag(v), K = exp(-lbd * M)
        u: tensor[m, n, 1] = self.u.T[:, :, None]
        v: tensor[m, 1, n] = self.v.T[:, None, :]
        # the first dimension of out corresponds to the number of target histograms
        out: tensor[m, n, n] = u * self.K[None, :, :] * v

        if u.shape[0] == 1:
            out: tensor[n, n] = out.squeeze()
        return out

    def solve(self, M, ldb, r, c):
        self.u, self.v, self.K = sinkhorn_knopp(M, ldb, r, c, self.niter, self.eps)


def sinkhorn_knopp(
    M: tensor["n n"],
    lbd: int,
    r: tensor["n 1"],
    C: tensor["n m"],
    niter: int | None = None,
    eps: float | None = 1e-4,
):
    """
    Solve the entropic OT problem using matrix scaling algorithm:
        P^{lbd} = argmin_{P} <P, M> - h(P)/lbd,
        P in transport polytope U(r, c)

    where
        M: A (n, n) Tensor, cost matrix
        ldb: the parameter of langrange multiplier lambda
        r: A (n, 1) Tensor, which is the vector of row sums (i.e. P @ ones(m) = r)
        C: A (n, m) Tensor, which is the collection of n-dimensional column sums (i.e. ones(n).T @ P = c.T)

    See: [Algorithm 1] in https://marcocuturi.net/Papers/cuturi13sinkhorn.pdf

    return:
        u: A (n, m) Tensor, row coefficients
        v: A (n, m) Tensor, column coefficients
        K: A (n, n) Tensor, exp(-lbd * M)
    """
    n = r.shape[0]
    m = C.shape[1]

    assert r.sum(dim=0) == 1, "The source histogram should sum to 1."
    assert (C.sum(dim=0) == torch.ones((1, m))).all(), "The target histograms should sum to 1."

    r: tensor[n] = r.squeeze()
    positive_i = r > 0
    r, M = r[positive_i], M[positive_i, :]

    K: tensor[n, n] = torch.exp(-lbd * M)

    old_d = torch.inf
    u: tensor[n, m] = torch.ones(n, m) / n
    v: tensor[n, m] = C / (K.T @ u)

    # Sinkhorn’s fixed point iteration (u, v) <- (r./Kv, c./K'u).
    if niter is not None:
        for _ in range(niter):
            u = r / (K @ v)
            v = C / (K.T @ u)
    else:
        while True:
            u = r / (K @ v)
            v = C / (K.T @ u)
            d = (u * ((K * M) @ v)).sum(dim=0)
            if torch.max(torch.abs(d - old_d)) < eps:
                break
            else:
                old_d = d
    return u, v, K
