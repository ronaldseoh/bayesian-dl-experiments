import torch.nn as nn
import torch.nn.functional as F

from .dropout_custom import create_dropout_layer


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
            self.dropout_rate, -1, self.dropout_type)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc1_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type)

        self.fc2 = nn.Linear(120, 84)

        self.fc2_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type)

        self.fc3 = nn.Linear(84, 10)
        
        self.fc3_dropout = create_dropout_layer(
            self.dropout_rate, -1, self.dropout_type)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_dropout(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_dropout(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1_dropout(self.fc1(x)))
        x = F.relu(self.fc2_dropout(self.fc2(x)))
        x = self.fc3_dropout(self.fc3(x))

        return x
