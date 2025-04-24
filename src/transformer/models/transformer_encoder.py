import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer encoder model

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
            dim_feedforward=r_dim * 4,
            dropout=0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=encoder_layers
        )

        self.out = nn.Linear(r_dim, y_dim)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, labels=None, mlm_mask=None, attention_mask=None):
        """
        Forward pass through the Transformer.

        Parameters
        ----------
        x : torch.Tensor of shape [B, seq_len, 1]
            Input token sequences.

        labels : torch.Tensor of shape [B, seq_len, 1] (optional)
            The original token values for masked positions (dummy values elsewhere).

        mlm_mask : torch.Tensor of shape [B, seq_len, 1] (optional)
            Binary mask indicating which positions were masked (1 for masked, 0 for unmasked).

        Returns
        -------
        dict
            If labels and mlm_mask are provided, returns a dict with 'loss' and 'logits'.
            Otherwise, returns a dict with 'logits'.
        """

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = self.project_r(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        logits = self.out(x).squeeze(-1)

        if labels is not None and mlm_mask is not None:
            # Compute loss only on masked tokens.
            loss = F.mse_loss(logits[mlm_mask == 1], labels[mlm_mask == 1])
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    # def forward(self, x_context, y_context, x_target):
    #     n_x_target = x_target.size(1)

    #     # [batch_size, 2 * n_context + x_target, x_dim]
    #     x = torch.cat([x_context, y_context, x_target], dim=1)

    #     # [batch_size, 2 * n_context + n_target, x_dim]
    #     R_c = self.project_r(x)
    #     R_c = self.pos_encoder(R_c)

    #     # [batch_size, 2 * n_context + n_target, x_dim]
    #     R = self.encoder(R_c)

    #     # [batch_size, n_target, y_dim]
    #     out = self.out(R[:, -n_x_target:, :])

    #     return out
