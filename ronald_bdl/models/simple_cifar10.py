import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import create_dropout_layer


class SimpleCIFAR10(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleCIFAR10, self).__init__()

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.conv1_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc1_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc2 = nn.Linear(120, 84)

        self.fc2_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc3 = nn.Linear(84, 10)

        self.fc3_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_dropout(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_dropout(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1_dropout(self.fc1(x)))
        x = F.relu(self.fc2_dropout(self.fc2(x)))
        x = self.fc3_dropout(self.fc3(x))

        return x

    def predict_dist(self, test_data, n_prediction):

        was_eval = not self.training

        predictions = []
        mean_predictions = []
        metrics = {}

        metrics['accuracy_mc'] = 0
        metrics['accuracy_non_mc'] = 0
        metrics['test_ll_mc'] = 0

        with torch.no_grad():
            if isinstance(test_data, torch.utils.data.DataLoader):
                for data in test_data:
                    # Temporaily disable eval mode
                    if was_eval:
                        self.train()

                    inputs, targets = data

                    # Determine where our test data needs to be sent to
                    # by checking the first conv layer weight's location
                    first_weight_location = self.conv1.weight.device

                    inputs = inputs.to(first_weight_location)
                    targets = targets.to(first_weight_location)

                    raw_scores_batch = torch.stack(
                        [self.forward(inputs) for _ in range(n_prediction)])

                    predictions_batch = torch.max(raw_scores_batch, 2).values

                    mean_raw_scores_batch = torch.mean(raw_scores_batch, 0)
                    mean_predictions_batch = torch.argmax(
                        mean_raw_scores_batch, 1)
                    mean_predictions.append(mean_predictions_batch)

                    if was_eval:
                        self.eval()

                    non_mc_raw_scores_batch = self.forward(inputs)
                    non_mc_predictions_batch = torch.argmax(
                        non_mc_raw_scores_batch, 1)

                    # Accuracy
                    metrics['accuracy_mc'] += torch.mean(
                        (mean_predictions_batch == targets).float())
                    metrics['accuracy_mc'] /= 2

                    # Accuracy (Non-MC)
                    metrics['accuracy_non_mc'] += torch.mean(
                        (non_mc_predictions_batch == targets).float())
                    metrics['accuracy_non_mc'] /= 2

                    # test log-likelihood
                    metrics['test_ll_mc'] -= (
                        F.cross_entropy(mean_raw_scores_batch, targets))
                    metrics['test_ll_mc'] /= 2

                mean_predictions = torch.cat(mean_predictions)

        return predictions, mean_predictions, metrics
