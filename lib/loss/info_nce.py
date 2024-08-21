import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _Loss


# Dimension hint for pytorch tensors
class tensor:
    def __getitem__(self, *args):
        return torch.Tensor


class InfoNCE(_Loss, nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction="mean", negative_mode="unpaired"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, temperature=None):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=temperature if temperature is not None else self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
        )


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction="mean", negative_mode="unpaired"):

    n, d = query.shape

    # Normalize to unit vectors
    query: tensor[n, d] = F.normalize(query, dim=-1)
    positive_key: tensor[n, d] = F.normalize(positive_key, dim=-1)
    if negative_keys is not None:
        m = negative_keys.shape[-2]
        negative_keys = F.normalize(negative_keys, dim=-1)

    # Compute InfoNCE
    if negative_keys is None:
        logits: tensor[n, n] = query @ positive_key.t()
        target: tensor[n] = torch.arange(n, device=query.device)

    elif negative_mode == "unpaired":
        positive_logit: tensor[n, 1] = torch.sum(query * positive_key, dim=1, keepdim=True)
        negative_logits: tensor[n, m] = query @ negative_keys.t()
        logits: tensor[n, m + 1] = torch.concat([positive_logit, negative_logits], dim=1)
        target: tensor[n] = torch.zeros(n, device=query.device).long()

    elif negative_mode == "paired":
        positive_logit: tensor[n, 1] = torch.sum(query * positive_key, dim=1, keepdim=True)
        negative_logits: tensor[n, m] = torch.einsum("N D, N M D -> N M", query, negative_keys)
        logits: tensor[n, m + 1] = torch.concat([positive_logit, negative_logits], dim=1)
        target: tensor[n] = torch.zeros(n, device=query.device).long()

    else:
        raise Exception("Invalid negative mode.")

    out = F.cross_entropy(logits / temperature, target, reduction=reduction)
    if out.isnan():
        breakpoint()
    return out
