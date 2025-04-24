import torch
import torch.nn as nn

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

    log_p = p_y_dist.log_prob(y_target).mean(-1)
    nll = -log_p

    if mask is not None:
        # Reshape mask to broadcast over extra dims
        while mask.dim() < nll.dim():
            mask = mask.unsqueeze(0)
        nll = nll * mask
        return nll.sum() / mask.sum().clamp_min(1.0)

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
