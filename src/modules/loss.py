import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.kl import kl_divergence


def negative_log_likelihood(p_y_dist, y_target, mask=None):
    """Computes the negative log likelihood of a distribution with target value y.

    Parameters
    ----------
    p_y_dist : torch.distributions.Distribution
        The predicted probability distribution over the target space.

    target_y : torch.Tensor
        The ground truth target values. The shape should be compatible with samples from `p_y_dist`.

    mask : torch.Tensor (optional)
        Float or bool mask of shape [batch, seq_len] (or broadcastable) with 1s
        indicating positions to include in the loss.

    Returns
    -------
    torch.Tensor
        The computed negative log likelihood.
    """

    log_p = p_y_dist.log_prob(y_target)

    if mask is not None:
        # Reshape mask to broadcast over extra dims
        while mask.dim() < log_p.dim():
            mask = mask.unsqueeze(0)
        log_p = log_p * mask
        log_p_sum = log_p.sum(dim=-1)
        valid_counts = mask.sum(dim=-1).clamp_min(1.0)
        avg_log_p = log_p_sum / valid_counts
    else:
        avg_log_p = log_p.mean(dim=-1)  # [n_z, batch, n_target]

    nll = -avg_log_p
    return nll.mean()


class NLLLoss(nn.Module):
    """Computes the negative log likelihood loss"""

    def __init__(self):
        super().__init__()

    def forward(self, p_y_dist, y_target, mask=None):
        return negative_log_likelihood(p_y_dist, y_target, mask=mask)


class ELBOLoss(nn.Module):
    """Computes the evidence lower bound loss"""

    def __init__(self):
        super().__init__()

    def forward(self, p_y_dist, q_zct, q_zc, y_target, mask=None):
        # L_vi = E[log(p_y_xc)] - kl(p_zct||p_zc)

        # [n_z, batch_size, n_targets]
        nll_loss = negative_log_likelihood(p_y_dist, y_target, mask=mask)

        kl_loss = kl_divergence(q_zct, q_zc)
        kl_loss = kl_loss.mean(dim=0).mean()

        return nll_loss + kl_loss


class MSELoss(nn.Module):
    """Computes the mean squared error loss

    Parameters
    ----------
    reduction : str (optional)
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        Default: 'mean'
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_target, mask=None):

        if mask is not None:
            mse = F.mse_loss(y_pred, y_target, reduction="none")
            mask = mask.bool()
            loss = mse[mask]
            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
        else:
            loss = F.mse_loss(y_pred, y_target, reduction=self.reduction)

        return loss
