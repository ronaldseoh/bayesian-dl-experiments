import torch.nn.functional as F
import torch.nn as nn

class FCNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden):
        super().__init__()

        self.n_hidden = n_hidden

        self.input = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList()

        for i in range(n_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_linear = nn.Linear(hidden_dim, output_dim)

        self.output = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.input(x)

        for i in range(self.n_hidden):
            x = self.hidden_layers[i](x)

        x = self.output_linear(x)

        x = self.output(x)

        return x