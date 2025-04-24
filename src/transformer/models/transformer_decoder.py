import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import PositionalEncoding


class TransformerDecoder(nn.Module):
    """Transformer decoder model

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    r_dim : int (optional)
        Dimension of the representation

    decoder_layers : int (optional)
        Number of decoder layers

    decoder_heads : int (optional)
        Number of decoder heads

    kwargs: dict
        Additional Transformer class arguments
        Passed to each TransformerEncoderLayer (e.g. dim_feedforward, dropout)
    """

    def __init__(
        self, x_dim, y_dim, r_dim=128, decoder_layers=2, decoder_heads=8, **kwargs
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.project_r = nn.Sequential(nn.Linear(x_dim, r_dim), nn.ReLU())
        self.pos_encoder = PositionalEncoding(r_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=r_dim,
            nhead=decoder_heads,
            dim_feedforward=r_dim * 4,
            **kwargs,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=decoder_layers
        )

        self.fc_out = nn.Linear(r_dim, y_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, sz):
        """
        Upper-triangular mask with True in forbidden positions.

        Parameters
        ----------
        sz: int
            Size of the mask (sequence length)
        """
        mask = torch.triu(
            torch.ones(sz, sz, device=next(self.parameters()).device), diagonal=1
        )
        return mask.bool()

    def forward(self, src, return_logits=True):
        """
        Parameters
        ----------
        src: torch.LongTensor
            Input sequence of shape (seq_len, batch, x_dim)

        return_logits: bool
            If True, return logits instead of probabilities

        Returns
        -------
        torch.FloatTensor
            Output sequence of shape (seq_len, batch, y_dim)
        """

        # Embed + Scale
        x = self.project_r(src) * math.sqrt(self.r_dim)  # (seq_len, batch, r_dim)
        x = self.pos_encoder(x)

        # Causal mask so positions can only see past & current
        seq_len = src.size(0)
        causal_mask = self._generate_causal_mask(seq_len)  # (seq_len, seq_len)

        # run through the stack of self-attention blocks
        hidden = self.transformer(x, mask=causal_mask)  # (seq_len, batch, r_dim)

        logits = self.fc_out(hidden)  # (seq_len, batch, y_dim)
        return logits if return_logits else torch.softmax(logits, dim=-1)
