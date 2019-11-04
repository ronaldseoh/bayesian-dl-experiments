import torch
import numpy as np

from .fcnet import FCNet

class FCNetMCDropout(FCNet):

    def __init__(
        self, input_dim, output_dim, hidden_dim, n_hidden, dropout_rate, n_predictions):
        super(FCNetMCDropout, self).__init__(
            input_dim, output_dim, hidden_dim, n_hidden, dropout_rate)

        self.n_predictions = n_predictions

    def mc_predict(self, X_test, **kwargs):
        # No gradient computation needed for predictions, mean, and var
        # Refer to https://pytorch.org/docs/stable/autograd.html#locally-disable-grad
        with torch.no_grad():
            predictions = torch.cat([self.forward(X_test) for _ in range(self.n_predictions)])

            mean = torch.mean(predictions, 0)
            var = torch.var(predictions, 0)

            # If y_test is given, calculate RMSE and test log-likelihood
            metrics = {}

            if 'y_test' in kwargs:
                y_test = kwargs['y_test']
                reg_strength = torch.tensor(kwargs['reg_strength'], dtype=torch.float)

                # RMSE
                metrics['rmse'] = torch.sqrt(torch.mean(torch.pow(y_test - mean, 2)))

                # test log-likelihood
                metrics['test_ll'] = torch.mean(
                    torch.logsumexp(- torch.tensor(0.5) * reg_strength * torch.pow(y_test - predictions, 2), 0)
                    - torch.log(torch.tensor(self.n_predictions, dtype=torch.float))
                    - torch.tensor(0.5) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float)) 
                    + torch.tensor(0.5) * torch.log(reg_strength)
                )

        return predictions, mean, var, metrics
