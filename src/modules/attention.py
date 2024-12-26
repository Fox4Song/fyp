import torch
from torch import nn


class Attention(nn.Module):
    """Attention Module

    Parameters
    ----------
    attention_type : str (optional)
        Type of attention to use.
    """

    def __init__(
        self,
        attention_type="dot",
        mask=None,
        dropout=0,
    ):
        super().__init__()

        self.mask = None

        self.dropout = nn.Dropout(p=dropout)

        if attention_type == "dot":
            self.attention = self.dot_attention

    def forward(self, k, q, v):
        return self.attention(k, q, v)

    def dot_attention(self, k, q, v):
        """Scaled Dot Product Attention"""

        kq_dim = q.size(-1) ** 0.5

        # [batch_size, n_key, n_query]
        attn_weights = torch.einsum("bkd,bqd->bqk", k, q) / kq_dim

        if self.mask is not None:
            attn_weights.masked_fill(self.mask == 0, -1e9)

        norm_attn_weights = torch.softmax(attn_weights, dim=-1)
        norm_attn_weights = self.dropout(norm_attn_weights)

        # [batch_size, n_key, v_dim]
        context = torch.bmm(norm_attn_weights, v)

        return context, norm_attn_weights
