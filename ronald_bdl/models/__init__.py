from torchvision.models import squeezenet1_1

from .fcnet import FCNet
from .fcnet_mc_dropout import FCNetMCDropout
from .fcnet_pyro import FCNetPyro

__all__ = (
    'FCNet',
    'FCNetMCDropout',
    'FCNetPyro',
    'squeezenet1_1',
)
