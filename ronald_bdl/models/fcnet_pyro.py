import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal, Uniform
from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

import numpy as np


def normal_like(X):
    # Looks like each scalar in X will be sampled with
    # this Normal distribution.
    return Normal(loc=0, scale=1).expand(X.shape)  # .to_event(X.dim())


class FCNetPyro(PyroModule):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, **kwargs):
        super().__init__()

        self.n_hidden = n_hidden

        if 'torch_device' in kwargs:
            self.device = kwargs['torch_device']

        # Setup layers
        # Input layer

        # Setting up distributions for self.input's weight and bias
        # Adapted from
        # https://forum.pyro.ai/t/converting-neural-network-to-pyro-and-prediction/1414/2
        self.input = PyroModule[nn.ModuleDict]({
            'linear': PyroModule[nn.Linear](input_dim, hidden_dim),
            'relu': PyroModule[nn.ReLU](),
        })

        self.input['linear'].weight = PyroSample(
            normal_like(self.input['linear'].weight))
        self.input['linear'].bias = PyroSample(normal_like(
            self.input['linear'].bias))

        self.input['linear'].weight = self.input['linear'].weight.to(self.device)
        self.input['linear'].bias = self.input['linear'].bias.to(self.device)

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = PyroModule[nn.ModuleList]()

            for i in range(n_hidden):
                hidden_layer = nn.ModuleDict({
                    'linear': PyroModule[nn.Linear](hidden_dim, hidden_dim),
                    'relu': PyroModule[nn.ReLU](),
                }).to(self.device)

                hidden_layer['linear'].weight = PyroSample(
                    normal_like(hidden_layer['linear'].weight))
                hidden_layer['linear'].bias = PyroSample(
                    normal_like(hidden_layer['linear'].bias))

                hidden_layer['linear'].weight = hidden_layer['linear'].weight.to(self.device)
                hidden_layer['linear'].bias = hidden_layer['linear'].bias.to(self.device)

                self.hidden_layers.append(hidden_layer)

        # Output
        self.output = PyroModule[nn.Linear](hidden_dim, output_dim).to(self.device)

        self.output.weight = PyroSample(normal_like(self.output.weight))
        self.output.bias = PyroSample(normal_like(self.output.bias))

        self.output.weight = self.output.weight.to(self.device)
        self.output.bias = self.output.bias.to(self.device)

        # Guide function for inference
        self.guide = AutoDiagonalNormal(self)

    def forward(self, X, y=None):

        # Forward through the first layer
        activation = self.input['linear'](X)
        activation = self.input['relu'](activation)

        # Forward through hidden layers
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = hidden['linear'](activation)
                activation = hidden['relu'](activation)

        output = self.output(activation).squeeze(-1)

        sigma = pyro.sample("sigma", Normal(0., 9.))

        sigma = sigma.to(self.device)

        with pyro.plate("data", X.shape[0], device=self.device):
            obs = pyro.sample("obs", Normal(output, sigma), obs=y)

        return output

    def predict_dist(self, X_test, n_predictions, **kwargs):
        predictive = Predictive(self, num_samples=n_predictions)

        predictions = predictive(X_test)['obs']

        mean = torch.mean(predictions, 0)
        var = torch.var(predictions, 0)

        # If y_test is given, calculate RMSE and test log-likelihood
        metrics = {}

        if 'y_test' in kwargs:
            y_test = kwargs['y_test']
            reg_strength = torch.tensor(
                kwargs['reg_strength'], dtype=torch.float)

            # RMSE
            metrics['rmse_mc'] = torch.sqrt(
                torch.mean(torch.pow(y_test - mean, 2)))

            # RMSE (Non-MC)
            prediction_non_mc = self.forward(X_test)
            metrics['rmse_non_mc'] = torch.sqrt(
                torch.mean(torch.pow(y_test - prediction_non_mc, 2)))

            # test log-likelihood
            metrics['test_ll_mc'] = torch.mean(
                torch.logsumexp(
                    - torch.tensor(0.5) * reg_strength * torch.pow(
                        y_test[None] - predictions, 2), 0)
                - torch.log(torch.tensor(n_predictions, dtype=torch.float))
                - torch.tensor(0.5) * torch.log(
                    torch.tensor(2 * np.pi, dtype=torch.float))
                + torch.tensor(0.5) * torch.log(reg_strength)
            )

        return predictions, mean, var, metrics
