from functools import partial

import torch
import torch.nn as nn

from modules import MLP

from .base import NeuralProcessFamily, LatentNeuralProcessFamily


class CNP(NeuralProcessFamily):
    """Conditional Neural Process

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    Encoder : nn.Module (optional)
        Encoder module

    kwargs: dict
        Additional Neural Process Family base class arguments
    """

    def __init__(self, x_dim, y_dim, Encoder=None, **kwargs):
        super().__init__(x_dim, y_dim, **kwargs)

        if Encoder is None:
            Encoder = partial(
                MLP,
                n_hidden_layers=7,
                hidden_size=self.r_dim,
                dropout=0.2,
                is_res=True,
            )
        self.encoder = Encoder(self.x_dim + self.y_dim, self.r_dim)

    def encode_context_representation(self, x_context, y_context):

        batch_size, n_context, x_dim = x_context.shape
        _, _, y_dim = y_context.shape

        # Concatenate x and y along the feature axis
        # [batch_size, n_context, x_dim + y_dim]
        encoder_input = torch.cat([x_context, y_context], dim=-1)

        # Flatten the context points
        # [batch_size * n_context, x_dim + y_dim]
        x = encoder_input.view(batch_size * n_context, -1)
        assert x.shape == (batch_size * n_context, x_dim + y_dim)

        # Encode the context into representation R
        # [batch_size * n_context, r_dim]
        x = self.encoder(x)
        x = x.view(batch_size, n_context, -1)

        # Mean aggregation
        # [batch_size, 1, r_dim]
        R = torch.mean(x, dim=1, keepdim=True)

        return R

    def encode_target_representation(self, x_context, z, R, x_target):

        _, n_target, _ = x_target.shape

        # Use same representation R for all target points
        # [batch_size, n_target, r_dim]
        R_target = R.repeat(1, n_target, 1)

        # n_z = 1
        # [1, batch_size, n_target, r_dim]
        R_target = R_target.unsqueeze(0)

        return R_target


class LNP(LatentNeuralProcessFamily, CNP):
    """Latent Neural Process

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    **kwargs: dict
        Additional Neural Process Family base class arguments
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        **kwargs,
    ):
        super().__init__(x_dim, y_dim, **kwargs)

    def encode_target_representation(self, x_context, z, R, x_target):

        _, n_target, _ = x_target.shape

        # Concatenate R and z along the feature axis
        # [n_z, batch_size, n_lat, r_dim]
        R_target = self.concat_r_z(R, z)

        # [n_z, batch_size, n_target, r_dim]
        R_target = R_target.repeat(1, 1, n_target, 1)

        return R_target
