"""Feed Forward Module"""

import torch.nn as nn


class FeedForward(nn.Module):
    """Feed Forward Module

    Parameters
    ----------
    dim : int
        Dimension of the input

    dropout : float (optional)
        Dropout rate
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.feed_forward(x)
