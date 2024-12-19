import torch.nn as nn


class MLP(nn.module):
    """General MLP Class

    Parameters
    ----------
    input_size : int
        Size of the input

    output_size : int
        Size of the output

    hidden_size : int
        Size of the hidden layers

    n_hidden_layers : int
        Number of hidden layers

    activation : torch.nn.Module
        Activation function to use

    is_bias : bool
        Whether to use bias in the linear layers

    dropout : float
        Dropout rate

    is_res : bool
        Whether to use residual connections
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.Relu(),
        is_bias=True,
        dropout=0.1,
        is_res=False,
    ):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._n_hidden_layers = n_hidden_layers
        self._activation = activation
        self._is_res = is_res

        self._dropout = nn.Dropout(p=dropout)

        self._to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self._linears = nn.ModuleList(
            [
                nn.Linear(self._hidden_size, self._hidden_size, bias=is_bias)
                for _ in range(self._n_hidden_layers - 1)
            ]
        )
        self._out = nn.Linear(self._hidden_size, self._output_size, bias=is_bias)

        # Initialise weights with Kaiming for RELU for now
        nn.init.kaiming_uniform_(self._to_hidden.weight, nonlinearity="relu")
        for linear in self._linears:
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self._out.weight, nonlinearity="relu")

    def forward(self, x):
        x = self._activation(self._to_hidden(x))
        res = self._dropout(x)

        for linear in self._linears:
            x = self._activation(linear(res))
            if self._is_res:
                x = x + res
            x = self._dropout(x)
            res = x

        x = self._out(res)
        return x
