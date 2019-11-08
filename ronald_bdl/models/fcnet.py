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
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            create_dropout_layer(
                self.dropout_rate, self.dropout_type, self.dropout_variational_dim),
        )

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = nn.ModuleList()
            for i in range(n_hidden):
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        create_dropout_layer(
                            self.dropout_rate, self.dropout_type, self.dropout_variational_dim),
                    )
                )

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        activation = F.relu(self.input(X))
 
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = F.relu(hidden(activation))

        return self.output(activation)
