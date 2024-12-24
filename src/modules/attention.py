import torch
from torch import nn


class Attention(nn.Module):

    def __init__(
        self,
        attention_type="dot",
    ):
        super().__init__()

        if attention_type == "dot":
            self.attention = self.dot_attention

    def forward(self, k, q, v):
        return self.attention(k, q, v)

    def dot_attention(self, k, q, v):

        kq_dim = q.size(-1) ** 0.5

        # [batch_size, n_key, n_query]
        attn_weights = torch.einsum("bkd,bqd->bqk", k, q) / kq_dim
        norm_attn_weights = torch.softmax(attn_weights, dim=-1)

        # [batch_size, n_key, v_dim]
        context = torch.bmm(norm_attn_weights, v)

        return context
