import torch
import torch.nn.functional as F
import numpy as np

from .simple_cifar10 import SimpleCIFAR10


class SimpleCIFAR10MCDropout(SimpleCIFAR10):

    def __init__(self, dropout_rate, dropout_type):
        super(SimpleCIFAR10MCDropout, self).__init__(
            dropout_rate=dropout_rate, dropout_type=dropout_type,)

    def predict_dist(self, test_loader, n_prediction, torch_device):

        was_eval = not self.training

        predictions = []
        mean_predictions = []
        metrics = {}

        metrics['accuracy_mc'] = 0
        metrics['accuracy_non_mc'] = 0
        metrics['test_ll_mc'] = 0

        with torch.no_grad():
            for data in test_loader:
                # Temporaily disable eval mode
                if was_eval:
                    self.train()

                inputs, targets = data

                inputs = inputs.to(torch_device)
                targets = targets.to(torch_device)

                raw_scores_batch = torch.stack(
                    [self.forward(inputs) for _ in range(n_prediction)])

                predictions_batch = torch.max(raw_scores_batch, 2).values

                mean_raw_scores_batch = torch.mean(raw_scores_batch, 0)
                mean_predictions_batch = torch.argmax(mean_raw_scores_batch, 1)
                mean_predictions.append(mean_predictions_batch)

                if was_eval:
                    self.eval()

                non_mc_raw_scores_batch = self.forward(inputs)
                non_mc_predictions_batch = torch.argmax(non_mc_raw_scores_batch, 1)

                # Accuracy
                metrics['accuracy_mc'] += torch.mean((mean_predictions_batch == targets).float())
                metrics['accuracy_mc'] /= 2

                # Accuracy (Non-MC)
                metrics['accuracy_non_mc'] += torch.mean((non_mc_predictions_batch == targets).float())
                metrics['accuracy_non_mc'] /= 2

                # test log-likelihood
                metrics['test_ll_mc'] -= (F.cross_entropy(mean_raw_scores_batch, targets))
                metrics['test_ll_mc'] /= 2

            mean_predictions = torch.cat(mean_predictions)

        return predictions, mean_predictions, metrics
