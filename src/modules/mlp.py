"""MLP module for Neural Process"""

import torch.nn as nn


class MLP(nn.Module):
    """General MLP Class

    Parameters
    ----------
    input_size : int
        Size of the input

    output_size : int
        Size of the output

    hidden_size : int (optional)
        Size of the hidden layers

    n_hidden_layers : int (optional)
        Number of hidden layers

    activation : torch.nn.Module (optional)
        Activation function to use

    is_bias : bool (optional)
        Whether to use bias in the linear layers

    dropout : float (optional)
        Dropout rate

    is_res : bool (optional)
        Whether to use residual connections
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=True,
        dropout=0.1,
        is_res=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.is_res = is_res

        self.dropout = nn.Dropout(p=dropout)

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)

        # Initialise weights with Kaiming for RELU for now
        nn.init.kaiming_uniform_(self.to_hidden.weight, nonlinearity="relu")
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.activation(self.to_hidden(x))
        res = self.dropout(x)

        for linear in self.linears:
            x = self.activation(linear(res))
            if self.is_res:
                x = x + res
            x = self.dropout(x)
            res = x

        x = self.out(res)
        return x