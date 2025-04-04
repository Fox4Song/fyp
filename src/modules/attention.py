import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP


class Attention(nn.Module):
    """Attention Module

    Implements either Scaled Dot-Product Attention or Multi-Head Attention mechanisms.

    Parameters
    ----------
    kq_size : int
        Size of the keys and queries.

    value_size : int
        Size of the values.

    out_size : int
        Size of the output to be projected after attention.

    n_heads : int (optional)
        Number of attention heads for Multi-Head Attention.
        Must divide `kq_size` and `value_size` evenly if `attention_type` is "multihead".

    attention_type : str (optional)
        Type of attention mechanism to use. Supported options (currently) are:
        - `"dot"`: Scaled Dot-Product Attention.
        - `"multihead"`: Multi-Head Attention.
        Default is `"dot"`.

    rep : str (optional)
        Representation type for processing keys and queries. Supported options are:
        - `"mlp"`: Apply separate MLPs to keys and queries before attention.
        - `"identity"`: Use keys and queries as-is without transformation.
        Default is `"mlp"`.

    x_size : int (optional)
        Input feature size for the MLP when `rep` is `"mlp"`.

    attention_layers : int (optional)
        Number of hidden layers in the MLP when `rep` is `"mlp"`.

    mask : torch.Tensor (optional)
        Mask tensor to apply to the attention weights. Typically used to mask out
        certain positions (e.g., padding or future tokens). Shape should be broadcastable
        with the attention weights.

    dropout : float (optional)
        Dropout probability to apply to the attention weights (only applicable for
        `"dot"` attention type).
    """

    def __init__(
        self,
        kq_size,
        v_size,
        out_size,
        x_size=1,
        n_heads=8,
        attention_type="dot",
        rep="mlp",
        attention_layers=2,
        mask=None,
        dropout=0,
    ):
        super().__init__()

        self.rep = rep
        self.mask = mask  # Expected shape: [batch_size, n_queries, n_keys]
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)

        if self.rep == "mlp":
            self.rep_k = MLP(
                x_size,
                kq_size,
                hidden_size=kq_size,
                n_hidden_layers=attention_layers,
                dropout=dropout,
            )
            self.rep_q = MLP(
                x_size,
                kq_size,
                hidden_size=kq_size,
                n_hidden_layers=attention_layers,
                dropout=dropout,
            )

        if attention_type == "dot":
            self.attention = self.dot_attention

        elif attention_type in ("multihead", "transformer"):
            # Ensure dimensions are divisible by the number of heads
            assert kq_size % n_heads == 0
            assert v_size % n_heads == 0
            if attention_type == "transformer":
                # For transformer attention, we require the output to match the key/query dimension
                assert kq_size == out_size
            self.attention = (
                self.multi_head_attention
                if attention_type == "multihead"
                else self.transformer_attention
            )
            self.kq_size = kq_size
            self.v_size = v_size
            self.out_size = out_size
            self.n_heads = n_heads
            self.kq_head_size = kq_size // n_heads
            self.v_head_size = v_size // n_heads

            # Linear projections for keys, queries, and values, and the output projection
            self.W_k = nn.Linear(kq_size, kq_size, bias=False)
            self.W_q = nn.Linear(kq_size, kq_size, bias=False)
            self.W_v = nn.Linear(v_size, v_size, bias=False)
            self.W_o = nn.Linear(v_size, out_size, bias=False)

            if attention_type == "transformer":
                self.layer_norm1 = nn.LayerNorm(out_size)
                self.layer_norm2 = nn.LayerNorm(out_size)
                # Feed-Forward Network with an expansion factor of 4
                self.ffn = MLP(out_size, out_size, hidden_size=4 * out_size)
            self._reset_parameters()

        else:
            raise NotImplementedError(
                f"Attention type '{attention_type}' is not implemented."
            )

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.W_k.weight)
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.xavier_normal_(self.W_v.weight)
        nn.init.xavier_normal_(self.W_o.weight)

    def forward(self, k, q, v):
        if self.rep != "identity":
            k = self.rep_k(k)
            q = self.rep_q(q)
        return self.attention(k, q, v)

    def _dot_attention(self, k, q, v):
        """Scaled Dot Product Attention

        Parameters
        ----------
        k : torch.Tensor [batch_size, n_k, kq_size]
            Keys in the attention mechanism

        q : torch.Tensor [batch_size, n_q, kq_size]
            Queries in the attention mechanism

        v : torch.Tensor [batch_size, n_v, v_size]
            Values in the attention mechanism

        Returns
        -------
        torch.Tensor [batch_size, n_query, v_size]
            Attention output
        """

        kq_dim = q.size(-1) ** 0.5

        # [batch_size, n_key, n_query]
        attn_weights = torch.einsum("bkd,bqd->bqk", k, q) / kq_dim

        if self.mask is not None:
            attn_weights.masked_fill(self.mask == 0, -1e9)

        norm_attn_weights = torch.softmax(attn_weights, dim=-1)
        norm_attn_weights = self.dropout(norm_attn_weights)

        # [batch_size, n_key, v_dim]
        rep = torch.bmm(norm_attn_weights, v)

        return rep

    def dot_attention(self, k, q, v):
        """Pytorch Scaled Dot-Product Attention with Flash Attention"""

        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask, dropout_p=self.dropout_p
        )

    def multi_head_attention(self, k, q, v):
        """Multihead Attention

        Parameters
        ----------

        k : torch.Tensor [batch_size, n_k, kq_size]
            Keys in the attention mechanism

        q : torch.Tensor [batch_size, n_q, kq_size]
            Queries in the attention mechanism

        v : torch.Tensor [batch_size, n_v, kq_size]
            Values in the attention mechanism

        Returns
        -------
        rep : torch.Tensor [batch_size, n_q, out_size]
            Multihead attention output
        """

        if self.mask is not None:
            self.mask.unsqueeze(1)

        batch_size = q.size(0)

        # Apply Linear Projections
        k = self.W_k(k)
        q = self.W_q(q)
        v = self.W_v(v)

        # [batch_size * n_heads, n_k, head_size]
        k, v, q = [
            (
                x.view(batch_size, -1, self.n_heads, self.kq_head_size)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size * self.n_heads, -1, self.kq_head_size)
            )
            for x in (k, v, q)
        ]

        # Apply scaled dot-product attention
        # [batch_size * n_heads, n_k, head_size]
        attn_head_weights = self.dot_attention(k, q, v)

        # Concat attention heads
        # [batch_size, n_k, v_size]
        multi_head_attn = (
            attn_head_weights.view(batch_size, self.n_heads, -1, self.kq_head_size)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.kq_head_size)
        )

        # Final Linear Projection
        # [batch_size, n_q, out_size]
        out = self.W_o(multi_head_attn)

        return out

    def transformer_attention(self, k, q, v):
        """Transformer Attention

        Parameters
        ----------
        k : torch.Tensor [batch_size, n_k, kq_size]
            Keys in the attention mechanism

        q : torch.Tensor [batch_size, n_q, kq_size]
            Queries in the attention mechanism

        v : torch.Tensor [batch_size, n_v, kq_size]
            Values in the attention mechanism

        Returns
        -------
        rep : torch.Tensor [batch_size, n_q, out_size]
            Transformer attention output
        """

        out = self.multi_head_attention(k, q, v)
        # Add & Norm
        out = self.layer_norm1(out + q)
        # Feed-forward network with residual connection and normalisation
        out = self.layer_norm2(out + self.ffn(out))

        return out
