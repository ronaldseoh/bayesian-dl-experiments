import torch
import numpy as np

from .fcnet import FCNet

class FCNetMCDropout(FCNet):

    def __init__(
        self, input_dim, output_dim, hidden_dim, n_hidden, 
        dropout_rate, dropout_type, dropout_variational_dim=None):
        super(FCNetMCDropout, self).__init__(
            input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_hidden=n_hidden, 
            dropout_rate=dropout_rate, dropout_type=dropout_type, dropout_variational_dim=dropout_variational_dim)

    def mc_predict(self, X_test, n_predictions, **kwargs):
        # No gradient computation needed for predictions, mean, and var
        # Refer to https://pytorch.org/docs/stable/autograd.html#locally-disable-grad
        with torch.no_grad():
            # Temporaily disable eval mode
            self.train()
            predictions = torch.stack([self.forward(X_test) for _ in range(n_predictions)])
            self.eval()

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
