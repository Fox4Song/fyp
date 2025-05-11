"""Base Modules for Neural Process Models."""

import abc
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP


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
                n_hidden_layers=7,
                hidden_size=self.r_dim,
                dropout=0.1,
                is_res=True,
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
        p_y_target : torch.distributions.Distribution [n_z, batch_size, n_target, y_dim]
            Posterior distribution for y valuees

        z : torch.Tensor [n_z, batch_size, n_lat, z_dim]
            Sampled latent variables

        q_zc : torch.distributions.Distribution [batch_size, n_lat, z_dim]
            Prior distribution for z conditioned on context

        q_zct : torch.distributions.Distribution [batch_size, n_lat, z_dim]
            Posterior distribution for z conditioned on context and target
            Returns `None` if `y_target` is not provided (e.g., during testing).
        """

        # Encode context set into representation R
        R = self.encode_context_representation(x_context, y_context)

        # Sample latent variables z
        z, q_zc, q_zct = self.encode_latent(x_context, y_context, R, x_target, y_target)

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

    def encode_latent(self, x_context, y_context, R, x_target, y_target):
        """Encodes the latent variable z.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        y_context : torch.Tensor [batch_size, n_context, y_dim]
            Context y values

        R : torch.Tensor [batch_size, n_reps, r_dim]
            Global representation of the context set

        x_target : torch.Tensor [batch_size, n_target, x_dim]
            Target x values

        y_target : torch.Tensor [batch_size, n_target, y_dim]
            Target y values

        Returns
        -------
        z : torch.Tensor [n_z, batch_size, n_lat, z_dim]
            Sampled latent variablea

        q_zc : torch.distributions.Distribution [batch_size, n_lat, z_dim]
            Prior distribution for z conditioned on context

        q_zct : torch.distributions.Distribution [batch_size, n_lat, z_dim]
            Posterior distribution for z conditioned on context and target
        """

        return None, None, None

    @abc.abstractmethod
    def encode_target_representation(self, x_context, z, R, x_target):
        """Encodes the target-dependent context set into a global representation R.

        Parameters
        ----------
        x_context : torch.Tensor [batch_size, n_context, x_dim]
            Context x values

        z : torch.Tensor [n_z, batch_size, n_lat, r_dim]
            Sample from the prior distribution

        R : torch.Tensor [batch_size, n_reps, r_dim]
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

        # Result from Le, Tuan Anh, et al. Empirical Evaluation of Neural Process Objectives. 2018.
        p_y_sigma = 0.01 + 0.99 * F.softplus(p_y_logsigma)

        # [n_z, batch_size, n_target, y_dim]
        p_y_target = torch.distributions.normal.Normal(p_y_mu, p_y_sigma)

        return p_y_target


class LatentNeuralProcessFamily(NeuralProcessFamily):
    """Base class for Latent Neural Process Family members.

    Parameters
    ----------
    n_z_train : int
        Number of latent variables to sample during training

    n_z_test : int
        Number of latent variables to sample during testing

    LatentEncoder : torch.nn.Module
        Latent encoder module
    """

    def __init__(
        self,
        *args,
        n_z_train=1,
        n_z_test=1,
        LatentEncoder=None,
        gp=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.z_dim = self.r_dim
        self.n_z_train = n_z_train
        self.n_z_test = n_z_test

        if LatentEncoder is None:
            LatentEncoder = partial(
                MLP,
                n_hidden_layers=6,
                hidden_size=self.z_dim,
                dropout=0.1,
                is_res=True,
            )
        self.latent_encoder = LatentEncoder(self.r_dim, self.z_dim * 2)

        self.reshape_r_z = nn.Linear(self.r_dim + self.z_dim, self.r_dim)

    def forward(self, *args, **kwargs):

        self.n_z = self.n_z_train if self.training else self.n_z_test

        return super().forward(*args, **kwargs)

    def encode_latent(self, x_context, y_context, R, x_target, y_target):

        q_zc = self.infer_latent_dist(R)

        if self.training and y_target is not None:
            R_target = self.encode_context_representation(x_target, y_target)
            q_zct = self.infer_latent_dist(R_target)
            # Sample z from posterior distribution q(z|c,t)
            sampling_dist = q_zct
        else:
            q_zct = None
            # Sample z from prior distribution p(z|c)
            sampling_dist = q_zc

        # [n_z, batch_size, n_lat, z_dim]
        z = sampling_dist.rsample([self.n_z])

        return z, q_zc, q_zct

    def infer_latent_dist(self, R):
        """Infer the latent distribution given global representation.

        Parameters
        ----------
        X : torch.Tensor [batch_size, n_lat, x_dim]
            Features to condition on

        R : torch.Tensor [batch_size, n_lat, r_dim]
            Global representation of the context set

        Returns
        -------
        q_zc : torch.distributions.Distribution [batch_size, n_lat, z_dim]
            Inferred latent distribution
        """

        # [batch_size, n_lat, z_dim]
        R = self.rep_to_lat_input(R)

        batch_size, n_lat, _ = R.shape

        # [batch_size * n_lat, r_dim]
        R_input = R.view(batch_size * n_lat, -1)
        # [batch_size * n_lat, z_dim * 2]
        q_z_stats = self.latent_encoder(R_input)
        # [batch_size, n_lat, z_dim * 2]
        q_z_stats = q_z_stats.view(batch_size, n_lat, -1)

        # [batch_size, n_lat, z_dim]
        q_z_mu, q_z_logsigma = q_z_stats.split(self.z_dim, dim=-1)

        # Taken from the deepmind repo
        q_z_sigma = 0.01 + 0.99 * F.sigmoid(0.5 * q_z_logsigma)
        # [batch_size, n_lat, z_dim]
        q_zc = torch.distributions.normal.Normal(q_z_mu, q_z_sigma)

        return q_zc

    def rep_to_lat_input(self, R):
        """Converts the representation into the latent input to be encoded.

        Parameters
        ----------
        R : torch.Tensor [batch_size, n_reps, r_dim]
            Representation of the context set

        Returns
        -------
        lat_input : torch.Tensor [batch_size, n_lat, lat_dim]
            Latent input to be encoded
        """

        return R

    def concat_r_z(self, R, z):
        """Concatenate the global representation R with the latent variable z.

        Parameters
        ----------
        R : torch.Tensor [batch_size, n_lat, r_dim]
            Global representation of the context set

        z : torch.Tensor [n_z, batch_size, n_lat, r_dim]
            Latent variable

        Returns
        -------
        R_z_reshaped : torch.Tensor [n_z, batch_size, n_lat, r_dim]
            Concatenated tensor reshaped to match the input shape of the decoder
        """

        assert R.size(1) == z.size(2)

        # [n_z, batch_size, n_lat, z_dim]
        R = R.unsqueeze(0).expand(z.size(0), -1, -1, -1)

        # [n_z, batch_size, n_lat, r_dim + z_dim]
        R_z = torch.cat([R, z], dim=-1)

        # Reshape to r_dim
        # [n_z, batch_size, n_lat, r_dim]
        R_z_reshaped = F.relu(self.reshape_r_z(R_z))

        return R_z_reshaped
