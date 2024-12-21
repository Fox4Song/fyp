"""Base Modules for Neural Process Models."""

import abc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_process.architectures.mlp import MLP


class NeuralProcessFamily(nn.Module, abc.ABC):
    """Base class for Neural Process Family members.

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    r_dim : int (optional)
        Dimension of the representation

    Decoder : nn.Module (optional)
        Decoder module
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        r_dim=128,
        Decoder=None,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        if Decoder is None:
            Decoder = partial(
                MLP,
                n_hidden_layers=2,
                hidden_size=self.r_dim,
            )
        self.decoder = Decoder(self.x_dim + self.r_dim, self.y_dim * 2)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """Returns the predicted mean and variance at the target points.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        y_context : torch.Tensor [batch_size, n_context, y_dim]
            Context y values

        x_target : torch.Tensor [batch_size, n_target, x_dim]
            Target x values

        y_target : torch.Tensor [batch_size, n_target, y_dim]
            Target y values

        Returns
        -------
        p_y_target : torch.distributions.Distribution
            Posterior distribution for y valuees

        z : torch.Tensor [batch_size, z_dim]
            Sampled latent variables

        q_zc : torch.distributions.Distribution
            Prior distribution for z conditioned on context

        q_zct : torch.distributions.Distribution
            Posterior distribution for z conditioned on context and target
        """

        # Encode context set into representation R
        R = self.encode_context_representation(x_context, y_context)

        # Sample latent variables z
        z, q_zc, q_zct = self.encode_latent(x_context, R, x_target, y_target)

        # Encode target-dependent representation R_target
        R_target = self.encode_target_representation(x_context, z, R, x_target)

        # Decode R_target into predictive distribution
        p_y_target = self.decode(x_target, R_target)

        return p_y_target, z, q_zc, q_zct

    @abc.abstractmethod
    def encode_context_representation(self, x_context, y_context):
        """Encodes the context set into a global representation R.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        y_context : torch.Tensor [batch_size, n_context, y_dim]
            Context y values

        Returns
        -------
        R : torch.Tensor [batch_size, n_reps, r_dim]
            Global representation of the context set
        """

        pass

    @abc.abstractmethod
    def encode_latent(self, x_context, R, x_target, y_target):
        """Encodes the latent variable z.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        R : torch.Tensor [batch_size, r_dim]
            Global representation of the context set

        x_target : torch.Tensor [batch_size, n_target, x_dim]
            Target x values

        y_target : torch.Tensor [batch_size, n_target, y_dim]
            Target y values

        Returns
        -------
        z : torch.Tensor [n_z, batch_size, n_latents, z_dim]
            Sampled latent variablea

        q_zc : torch.distributions.Distribution
            Prior distribution for z conditioned on context

        q_zct : torch.distributions.Distribution
            Posterior distribution for z conditioned on context and target
        """

        pass

    @abc.abstractmethod
    def encode_target_representation(self, x_context, z, R, x_target):
        """Encodes the target-dependent context set into a global representation R.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        z : torch.Tensor [batch_size, r_dim]
            Sample from the prior distribution

        R : torch.Tensor [batch_size, r_dim]
            Global representation of the context set

        x_target : torch.Tensor [batch_size, n_target, x_dim]
            Target x values

        Returns
        -------
        R : torch.Tensor [n_z, batch_size, n_targets, r_dim]
            Global representation of the target set
        """

        pass

    def decode(self, x_target, R_target):
        """Decode the target-dependent representation into a predictive distribution.

        Parameters
        ----------
        x_target : torch.Tensor [batch_size, n_target, x_dim]
            Target x values

        R_target : torch.Tensor [n_z, batch_size, n_target, r_dim]
            Global representation of the target set

        Returns
        -------
        p_y_target : torch.distributions.Distribution [n_z, batch_size, n_target, y_dim]
            Predictive distribution for target y values
        """

        batch_size, n_target, x_dim = x_target.shape
        n_z, _, _, r_dim = R_target.shape

        # [n_z, batch_size, n_target, x_dim]
        x_target_expanded = x_target.unsqueeze(0).repeat(n_z, 1, 1, 1)

        # Concatenate target x with R_target
        # [n_z, batch_size, n_target, x_dim + r_dim]
        decoder_input = torch.cat([x_target_expanded, R_target], dim=-1)

        # Flatten the batch and z dimensions
        # [n_z * batch_size * n_target, x_dim + r_dim]
        x = decoder_input.view(n_z * batch_size * n_target, -1)
        assert x.shape == (n_z * batch_size * n_target, x_dim + r_dim)

        # Decode into mean and variance
        # [n_z * batch_size * n_target, y_dim * 2]
        p_y_stats = self.decoder(x)
        # [n_z, batch_size, n_target, y_dim * 2]
        p_y_stats = p_y_stats.view(n_z, batch_size, n_target, -1)
        # [n_z, batch_size, n_target, y_dim]
        p_y_mu, p_y_logsigma = p_y_stats.split(self.y_dim, dim=-1)
        p_y_sigma = 0.01 + 0.99 * F.softplus(p_y_logsigma)

        # [n_z, batch_size, n_target, y_dim]
        p_y_target = torch.distributions.normal.Normal(p_y_mu, p_y_sigma)

        return p_y_target
