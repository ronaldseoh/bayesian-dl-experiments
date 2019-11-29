import torch
import torch.nn as nn

# Return a desired type of dropout layer or just identity layer to fit into
# dropout's places in our network implementations.
def create_dropout_layer(dropout_rate, dropout_variational_dim=None, dropout_type='identity'):

    if dropout_type == 'bernoulli':
        dropout_layer = nn.Dropout(dropout_rate)
    elif dropout_type == 'variational':
        dropout_layer = VariationalDropout(dropout_variational_dim, dropout_rate)
    else:
        # No dropout at all
        dropout_layer = nn.Identity()

    return dropout_layer
