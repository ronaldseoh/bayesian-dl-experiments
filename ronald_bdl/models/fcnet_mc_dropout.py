import torch

from .fcnet import FCNet

class FCNetMCDropout(FCNet):

    def __init__(
        self, input_dim, output_dim, hidden_dim, n_hidden, dropout_rate, n_predictions):
        super(FCNetMCDropout, self).__init__(
            input_dim, output_dim, hidden_dim, n_hidden, dropout_rate)

        self.n_predictions = n_predictions

    def mc_predict(self, x_test):
        predictions = torch.cat([self.forward(x_test) for _ in range(self.n_predictions)])

        mean = torch.mean(predictions, 0)
        var = torch.var(predictions, 0)

        return predictions, mean, var
