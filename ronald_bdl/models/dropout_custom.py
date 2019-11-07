import torch
import torch.nn as nn

# Ported from
# https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)
        
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = torch.Variable(torch.randn(x.size()))

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x