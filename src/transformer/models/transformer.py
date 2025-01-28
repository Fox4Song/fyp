import torch
import torch.nn as nn

from modules import PositionalEncoding


class Transformer(nn.Module):
    """Transformer model

    Parameters
    ----------
    x_dim : int
        Dimension of the x values

    y_dim : int
        Dimension of the y values

    r_dim : int (optional)
        Dimension of the representation

    encoder_layers : int (optional)
        Number of encoder layers

    encoder_heads : int (optional)
        Number of encoder heads

    kwargs: dict
        Additional Transformer class arguments
    """

    def __init__(
        self, x_dim, y_dim, r_dim=128, encoder_layers=2, encoder_heads=8, **kwargs
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

        self.project_r = nn.Sequential(nn.Linear(x_dim, r_dim), nn.ReLU())
        self.pos_encoder = PositionalEncoding(r_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=r_dim,
            nhead=encoder_heads,
            dim_feedforward=r_dim,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=encoder_layers
        )

        self.out = nn.Linear(r_dim, y_dim)

    def forward(self, x_context, y_context, x_target):
        n_x_target = x_target.size(1)
        print(n_x_target)
        # [batch_size, 2 * n_context + x_target, x_dim]
        x = torch.cat([x_context, y_context, x_target], dim=1)

        # [batch_size, 2 * n_context + n_target, x_dim]
        R_c = self.project_r(x)
        R_c = self.pos_encoder(R_c)

        # [batch_size, 2 * n_context + n_target, x_dim]
        R = self.encoder(R_c)

        # [batch_size, n_target, y_dim]
        print(R.shape)
        out = self.out(R[:, -n_x_target:, :])
        print(out.shape)
        return out
