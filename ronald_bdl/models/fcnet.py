import torch.nn.functional as F
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

            if self.dropout_type == 'variational':
                self.dropout_variational_dim = kwargs['dropout_variational_dim']
            else:
                self.dropout_variational_dim = None
    
        # Setup layers
        # Input layer
        self.input = nn.Linear(input_dim, hidden_dim)

        # Initialize weights
        nn.init.kaiming_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)

        # Hidden Layer(s)
        self.hidden_layers = nn.ModuleList()

        if n_hidden > 0:
            for i in range(n_hidden):
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                nn.init.kaiming_uniform_(self.hidden_layers[i].weight)
                nn.init.zeros_(self.hidden_layers[i].bias)

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)        

    def forward(self, X):
        dropout_input = create_dropout_layer(self.dropout_rate, self.dropout_type, self.dropout_variational_dim)
        activation = F.relu(dropout_input(self.input(X)))

        for hidden in self.hidden_layers:
            dropout_hidden = create_dropout_layer(self.dropout_rate, self.dropout_type, self.dropout_variational_dim)
            activation = F.relu(dropout_hidden(hidden(activation)))

        return self.output(activation)
