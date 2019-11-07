import torch.nn.functional as F
import torch.nn as nn

class FCNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, **kwargs):
        super(FCNet, self).__init__()

        self.n_hidden = n_hidden

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']

            if self.dropout_type == 'bernoulli':
                self.dropout_layer = nn.Dropout(self.dropout_rate)
            elif self.dropout_type == 'variational':
                from .dropout_custom import VariationalDropout
                self.dropout_variational_dim = kwargs['dropout_variational_dim']
                self.dropout_layer = VariationalDropout(self.dropout_rate, self.dropout_variational_dim)
        else:
            # No dropout at all
            self.dropout_layer = nn.Identity()

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
        activation = F.relu(self.dropout_layer(self.input(X)))

        for hidden in self.hidden_layers:
            activation = F.relu(self.dropout_layer(hidden(activation)))

        return self.output(activation)
