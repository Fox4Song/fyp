import math

import torch
import torch.nn as nn

# TODO: Positional and Relative encoding


class PositionalEncoding(nn.Module):
    """PE function. Taken from The Annotated Tranformer https://nlp.seas.harvard.edu/annotated-transformer/

    Parameters
    ----------
    d_model : int
        Dimension of the model

    dropout : float (optional)
        Dropout rate

    max_len : int (optional)
        Maximum length of the sequence
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
