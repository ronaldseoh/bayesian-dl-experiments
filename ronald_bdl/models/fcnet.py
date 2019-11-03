import torch.nn.functional as F
import torch.nn as nn

class FCNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, dropout_rate):
        super(FCNet, self).__init__()

        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        # Setup layers
        # Input layer
        self.input = nn.Linear(input_dim, hidden_dim)

        # Hidden Layer(s)
        self.hidden_layers = nn.ModuleList()

        if n_hidden > 0:
            for _ in range(n_hidden):
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        activation = F.relu(F.dropout(self.input(x), p=self.dropout_rate))

        for hidden in self.hidden_layers:
            activation = F.relu(F.dropout(hidden(activation), p=self.dropout_rate))

        return self.output(activation)
