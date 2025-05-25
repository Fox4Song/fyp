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
        activation=nn.LeakyReLU(0.1),
        is_bias=True,
        dropout=0,
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
        self.ln_in = nn.LayerNorm(self.hidden_size)
        
        self.linears = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(self.n_hidden_layers - 1):
            self.linears.append(nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias))
            self.lns.append(nn.LayerNorm(self.hidden_size))

        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialise weights with Kaiming for RELU for now
        nn.init.kaiming_uniform_(self.to_hidden.weight, nonlinearity="leaky_relu")
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")

    def forward(self, x):
        x = self.to_hidden(x)
        x = self.ln_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        res = x
        
        for linear in self.linears:
            x = self.activation(linear(res))
            if self.is_res:
                x = x + res
            x = self.dropout(x)
            res = x

        x = self.out(res)
        return x
