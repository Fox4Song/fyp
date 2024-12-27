import torch
import torch.nn as nn

from torch.distributions.kl import kl_divergence


def negative_log_likelihood(p_y_dist, y_target):
    """Computes the negative log likelihood of a distribution with target value y.

    Parameters
    ----------
    p_y_dist : torch.distributions.Distribution
        The predicted probability distribution over the target space.

    target_y : torch.Tensor
        The ground truth target values. The shape should be compatible with samples from `p_y_dist`.

    Returns
    -------
    torch.Tensor
        The computed negative log likelihood.
    """

    log_p = p_y_dist.log_prob(y_target).mean(-1)
    return -log_p


class NLLLoss(nn.Module):
    """Computes the negative log likelihood loss"""

    def __init__(self):
        super().__init__()

    def forward(self, p_y_dist, y_target):
        nll = negative_log_likelihood(p_y_dist, y_target)
        # Reduce with mean aggregation (alternatives: no aggregation, sum)
        return nll.mean()


class ELBOLoss(nn.Module):
    """Computes the evidence lower bound loss"""

    def __init__(self):
        super().__init__()

    def forward(self, p_y_dist, q_zct, q_zc, y_target):
        # L_vi = log(p_y_xc) - kl(p_zct||p_zc)

        # [n_z, batch_size, n_targets]
        nll_loss = negative_log_likelihood(p_y_dist, y_target)
        # [batch_size, n_targets]
        nll_loss = nll_loss.mean(0)

        # n_lat = 1.
        # [batch_size, n_lat]
        kl_loss = kl_divergence(q_zct, q_zc).mean(-1)
        kl_loss = kl_loss.expand(nll_loss.shape)

        loss = nll_loss + kl_loss
        return loss.mean()
