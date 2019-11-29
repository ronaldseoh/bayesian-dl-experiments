import torch
import numpy as np

from .simple_cifar10 import SimpleCIFAR10


class SimpleCIFAR10MCDropout(SimpleCIFAR10):

    def __init__(self, dropout_rate, dropout_type):
        super(SimpleCIFAR10MCDropout, self).__init__(
            dropout_rate=dropout_rate, dropout_type=dropout_type,)

    def predict_dist(self, test_loader, n_prediction, torch_device):

        was_eval = not self.training

        raw_scores = []
        raw_scores_non_mc = []
        
        for data in test_loader:
            # Temporaily disable eval mode
            if was_eval:
                self.train()

            inputs, targets = data
            
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)

            raw_scores_batch = torch.stack(
                [self.forward(inputs) for _ in range(n_prediction)])

            raw_scores.append(raw_scores_batch)
            
            if was_eval:
                self.eval()
                
            raw_scores_batch_non_mc = self.forward(inputs)
            
            raw_scores_non_mc.append(raw_scores_batch_non_mc)
            
        raw_scores = torch.stack(raw_scores)
        raw_scores_non_mc = torch.stack(raw_scores_non_mc)

        print(raw_scores.shape)
        print(raw_scores_non_mc.shape)
        
        mean = torch.mean(raw_scores, 1)
        var = torch.var(raw_scores, 1)
        
        print(mean.shape)

        # If y_test is given, calculate RMSE and test log-likelihood
        metrics = {}

        # Accuracy
        metrics['accuracy_mc'] = torch.mean(torch.max(mean, 1) == y_test)

        # Accuracy (Non-MC)
        metrics['accuracy_non_mc'] = torch.mean(
            torch.max(prediction_non_mc, 1) == y_test)

        return predictions, mean, var, metrics
