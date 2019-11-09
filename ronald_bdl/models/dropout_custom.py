import torch
import torch.nn as nn

# Ported from
# https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
class VariationalDropout(nn.Module):
    def __init__(self, dim, dropout_rate=0):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = (1 - dropout_rate)
        
        # Initial alpha
        self.log_alpha = nn.Parameter(torch.log(torch.ones(dim) * self.max_alpha))

        # N(0,1)
        self.epsilon = nn.Parameter(torch.randn(dim))

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * torch.pow(alpha, 2) + c3 * torch.pow(alpha, 3)

        kl = - negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        if self.train():
            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)

            # N(1, alpha)
            self.epsilon.data = torch.mul(self.epsilon, self.log_alpha.exp())

            return torch.mul(x, self.epsilon)
        else:
            return x

def create_dropout_layer(dropout_rate, dropout_variational_dim=None, dropout_type='identity'):

    if dropout_type == 'bernoulli':
        dropout_layer = nn.Dropout(dropout_rate)
    elif dropout_type == 'variational':
        dropout_layer = VariationalDropout(dropout_variational_dim, dropout_rate)
    else:
        # No dropout at all
        dropout_layer = nn.Identity()

    return dropout_layer
