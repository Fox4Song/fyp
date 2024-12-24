from functools import partial

import torch
import torch.nn as nn

from modules import MLP, Attention

from .base import NeuralProcessFamily, LatentNeuralProcessFamily


class AttnCnp(NeuralProcessFamily):
    """Attentive Conditional Neural Process

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    Encoder : nn.Module (optional)
        Encoder module

    attention_type : str (optional)
        Type of attention to use.

    use_self_attention: bool (optional)
        Whether to use self attention in the encoder

    kwargs: dict
        Additional Neural Process Family base class arguments
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        Encoder=None,
        attention_type="dot",
        use_self_attention=True,
        **kwargs,
    ):
        super().__init__(x_dim, y_dim, **kwargs)

        self.use_self_attention = use_self_attention

        if Encoder is None:
            Encoder = partial(
                MLP,
                n_hidden_layers=3,
                hidden_size=self.r_dim,
            )
        self.encoder = Encoder(self.x_dim + self.y_dim, self.r_dim)

        if self.use_self_attention:
            self.self_attention = Attention(attention_type)

        self.cross_attention = Attention(attention_type)

    def encode_context_representation(self, x_context, y_context):

        # [batch_size, n_context, x_dim + y_dim]
        x = torch.cat([x_context, y_context], dim=-1)

        # [batch_size, n_context, r_dim]
        R_c = self.encoder(x)

        if self.use_self_attention:
            R_c = self.self_attention(R_c, R_c, R_c)

        return R_c

    def encode_target_representation(self, x_context, _, R, x_target):

        # [batch_size, n_target, r_dim]
        R_target = self.cross_attention(x_context, x_target, R)  # key, query, value

        # n_z = 1
        # [1, batch_size, n_target, r_dim]
        R_target = R_target.unsqueeze(0)

        return R_target
