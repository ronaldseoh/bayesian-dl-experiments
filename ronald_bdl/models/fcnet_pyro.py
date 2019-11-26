import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal, Uniform
import numpy as np

from .dropout_custom import create_dropout_layer


class FCNetPyro(pyro.nn.PyroModule):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, **kwargs):
        super().__init__()

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
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            create_dropout_layer(
                self.dropout_rate, hidden_dim, self.dropout_type,),
        )

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = nn.ModuleList()
            for i in range(n_hidden):
                self.hidden_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        create_dropout_layer(
                            self.dropout_rate, hidden_dim, self.dropout_type,),
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


    def predict_dist(self, X_test, n_predictions, **kwargs):
        sampled_models = [self.guide(None, None) for _ in range(n_predictions)]
        yhats = [model(X_test).data for model in sampled_models]
        predictions = torch.stack(yhats)

        mean = torch.mean(predictions, 0)
        var = torch.var(predictions, 0)

        # If y_test is given, calculate RMSE and test log-likelihood
        metrics = {}

        if 'y_test' in kwargs:
            y_test = kwargs['y_test']
            reg_strength = torch.tensor(kwargs['reg_strength'], dtype=torch.float)

            # RMSE
            metrics['rmse_mc'] = torch.sqrt(torch.mean(torch.pow(y_test - mean, 2)))

            # RMSE (Non-MC)
            prediction_non_mc = self.forward(X_test)
            metrics['rmse_non_mc'] = torch.sqrt(torch.mean(torch.pow(y_test - prediction_non_mc, 2)))

            # test log-likelihood
            metrics['test_ll_mc'] = torch.mean(
                torch.logsumexp(- torch.tensor(0.5) * reg_strength * torch.pow(y_test[None] - predictions, 2), 0)
                - torch.log(torch.tensor(n_predictions, dtype=torch.float))
                - torch.tensor(0.5) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float)) 
                + torch.tensor(0.5) * torch.log(reg_strength)
            )

        return predictions, mean, var, metrics
