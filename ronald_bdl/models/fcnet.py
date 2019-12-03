import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .utils import create_dropout_layer, create_nonlinearity_layer


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

        # Nonlinear layer setting
        if 'nonlinear_type' in kwargs:
            self.nonlinear_type = kwargs['nonlinear_type']
        else:
            self.nonlinear_type = 'relu'

        # Setup layers
        # Input layer
        self.input = nn.ModuleDict({
            'linear': nn.Linear(input_dim, hidden_dim),
            'dropout': create_dropout_layer(
                self.dropout_rate, self.dropout_type),
            'nonlinear': create_nonlinearity_layer(self.nonlinear_type),
        })

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = nn.ModuleList()

            for i in range(n_hidden):
                self.hidden_layers.append(
                    nn.ModuleDict({
                        'linear': nn.Linear(hidden_dim, hidden_dim),
                        'dropout': create_dropout_layer(
                            self.dropout_rate, self.dropout_type),
                        'nonlinear': create_nonlinearity_layer(
                            self.nonlinear_type),
                    })
                )

        # Output
        self.output = nn.ModuleDict({
            'linear': nn.Linear(hidden_dim, output_dim),
            'dropout': create_dropout_layer(
                self.dropout_rate, self.dropout_type),
        })

    def forward(self, X):
        # Forward through the input layer
        activation = self.input['linear'](X)
        activation = self.input['dropout'](activation)
        activation = self.input['nonlinear'](activation)

        # Forward through hidden layers
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = hidden['linear'](activation)
                activation = hidden['dropout'](activation)
                activation = hidden['nonlinear'](activation)

        activation = self.output['linear'](activation)
        activation = self.output['dropout'](activation)

        return activation

    def predict_dist(self, test_data, n_prediction=1000, **kwargs):

        was_eval = not self.training

        if 'y_mean' in kwargs:
            y_mean = kwargs['y_mean']
            y_std = kwargs['y_std']
        else:
            y_mean = 0
            y_std = 1

        # No gradient tracking necessary
        with torch.no_grad():
            if isinstance(test_data, torch.utils.data.DataLoader):

                predictions = []
                mean = 0

                metrics['rmse_mc'] = 0
                metrics['rmse_non_mc'] = 0
                metrics['test_ll_mc'] = 0

                reg_strength = torch.tensor(
                    kwargs['reg_strength'], dtype=torch.float)

                for data in test_data:
                    # Temporaily disable eval mode
                    if was_eval:
                        self.train()

                    inputs, targets = data

                    # Determine where our test data needs to be sent to
                    # by checking the first conv layer weight's location
                    first_weight_location = self.input['linear'].weight.device

                    inputs = inputs.to(first_weight_location)
                    targets = targets.to(first_weight_location)

                    predictions_batch = torch.stack(
                        [self.forward(inputs) for _ in range(n_prediction)])

                    mean_batch = torch.mean(predictions_batch, 0)

                    mean += mean_batch
                    mean /= 2

                    predictions.append(predictions_batch)

                    # RMSE
                    metrics['rmse_mc'] += torch.mean(
                        torch.pow(target - mean_batch, 2))
                    metrics['rmse_mc'] /= 2

                    # RMSE (Non-MC)
                    prediction_non_mc = self.forward(X_test)
                    prediction_non_mc = prediction_non_mc * y_std + y_mean

                    metrics['rmse_non_mc'] += torch.mean(
                        torch.pow(target - prediction_non_mc, 2))
                    metrics['rmse_non_mc'] /= 2

                    # test log-likelihood
                    metrics['test_ll_mc'] -= torch.mean(
                        torch.logsumexp(
                            - torch.tensor(0.5) * reg_strength * torch.pow(
                                y_test[None] - predictions, 2), 0)
                        - torch.log(
                            torch.tensor(n_predictions, dtype=torch.float))
                        - torch.tensor(0.5) * torch.log(
                            torch.tensor(2 * np.pi, dtype=torch.float))
                        + torch.tensor(0.5) * torch.log(reg_strength)
                    )
                    metrics['test_ll_mc'] /= 2

                predictions = torch.cat(predictions)
                var = torch.var(predictions)
                metrics['rmse_mc'] = torch.sqrt(metrics['rmse_mc'])
                metrics['rmse_non_mc'] = torch.sqrt(metrics['rmse_non_mc'])

            else:
                # Temporaily disable eval mode
                if was_eval:
                    self.train()

                predictions = torch.stack(
                    [self.forward(test_data) for _ in range(n_prediction)])

                predictions = predictions * y_std + y_mean

                if was_eval:
                    self.eval()

                mean = torch.mean(predictions, 0)
                var = torch.var(predictions, 0)

                # If y_test is given, calculate RMSE and test log-likelihood
                metrics = {}

                if 'y_test' in kwargs:
                    y_test = kwargs['y_test']
                    y_test = y_test * y_std + y_mean

                    reg_strength = torch.tensor(
                        kwargs['reg_strength'], dtype=torch.float)

                    train_size = kwargs['train_size']

                    # RMSE
                    metrics['rmse_mc'] = torch.sqrt(
                        torch.mean(torch.pow(y_test - mean, 2)))

                    # RMSE (Non-MC)
                    prediction_non_mc = self.forward(test_data)
                    prediction_non_mc = prediction_non_mc * y_std + y_mean
                    metrics['rmse_non_mc'] = torch.sqrt(
                        torch.mean(torch.pow(y_test - prediction_non_mc, 2)))

                    # test log-likelihood
                    tau = torch.tensor(
                        (1 - self.dropout_rate) * np.power(1e-2, 2)
                        / (2 * train_size * reg_strength))

                    metrics['test_ll_mc'] = torch.mean(
                        torch.logsumexp(
                            - torch.tensor(0.5) * tau * torch.pow(
                                y_test[None] - predictions, 2), 0)
                        - torch.log(
                            torch.tensor(n_prediction, dtype=torch.float))
                        - torch.tensor(0.5) * torch.log(
                            torch.tensor(2 * np.pi, dtype=torch.float))
                        + torch.tensor(0.5) * torch.log(tau)
                    )

        return predictions, mean, var, metrics
