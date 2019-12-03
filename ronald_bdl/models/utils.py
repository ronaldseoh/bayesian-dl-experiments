import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Return a desired type of dropout layer or just identity layer to fit into
# dropout's places in our network implementations.
def create_dropout_layer(dropout_rate, dropout_type='identity'):
    if dropout_type == 'bernoulli':
        dropout_layer = nn.Dropout(dropout_rate)
    else:
        # No dropout at all
        dropout_layer = nn.Identity()

    return dropout_layer


def create_nonlinearity_layer(nonlinear_type='relu'):
    if nonlinear_type == 'relu':
        return nn.ReLU()
    elif nonlinear_type == 'tanh':
        return nn.Tanh()
    elif nonlinear_type == 'sigmoid':
        return nn.Sigmoid()


def create_nonlinearity_layer_functional(nonlinear_type='relu'):
    if nonlinear_type == 'relu':
        return F.relu
    elif nonlinear_type == 'tanh':
        return F.tanh
    elif nonlinear_type == 'sigmoid':
        return F.sigmoid


def tau(dropout_rate, length_scale, train_size, reg_strength):
    tau = torch.tensor(
        np.power(length_scale, 2) * (1 - dropout_rate)
        / (2 * train_size * reg_strength))

    return tau


def reg_strength(dropout_rate, length_scale, train_size, tau):
    reg_strength = torch.tensor(
        np.power(length_scale, 2) * (1 - dropout_rate)
        / (2 * train_size * tau))

    return reg_strength
