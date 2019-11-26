import torch.nn as nn

from .dropout_custom import create_dropout_layer


class FCNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, **kwargs):
        super(FCNet, self).__init__()

        self.n_hidden = n_hidden

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        # Setup layers
        # Input layer
        self.input = nn.ModuleDict({
            'linear': nn.Linear(input_dim, hidden_dim),
            'dropout': create_dropout_layer(
                self.dropout_rate, hidden_dim, self.dropout_type,),
            'relu': nn.ReLU(),
        })

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = nn.ModuleList()

            for i in range(n_hidden):
                self.hidden_layers.append(
                    nn.ModuleDict({
                        'linear': nn.Linear(hidden_dim, hidden_dim),
                        'dropout': create_dropout_layer(
                            self.dropout_rate, hidden_dim, self.dropout_type,),
                        'relu': nn.ReLU(),
                    })
                )

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # Forward through the input layer
        activation = self.input['linear'](X)
        activation = self.input['dropout'](activation)
        activation = self.input['relu'](activation)

        # Forward through hidden layers
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = hidden['linear'](activation)
                activation = hidden['dropout'](activation)
                activation = hidden['relu'](activation)

        return self.output(activation)
