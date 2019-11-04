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

        # Initialize weights
        nn.init.normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)

        # Hidden Layer(s)
        self.hidden_layers = nn.ModuleList()

        if n_hidden > 0:
            for i in range(n_hidden):
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                nn.init.normal_(self.hidden_layers[i].weight)
                nn.init.zeros_(self.hidden_layers[i].bias)

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)        

    def forward(self, X):
        activation = F.relu(F.dropout(self.input(X), p=self.dropout_rate))

        for hidden in self.hidden_layers:
            activation = F.relu(F.dropout(hidden(activation), p=self.dropout_rate))

        return self.output(activation)
