from .fcnet import FCNet
from .fcnet_mc_dropout import FCNetMCDropout
from .fcnet_pyro import FCNetPyro
from .squeezenet_original import SqueezeNet
from .squeezenet_dropout import SqueezeNetDropout
from .simple_cifar10 import SimpleCIFAR10
from .simple_cifar10_mc_dropout import SimpleCIFAR10MCDropout

__all__ = (
    FCNet,
    FCNetMCDropout,
    FCNetPyro,
    SqueezeNet,
    SqueezeNetDropout,
    SimpleCIFAR10,
    SimpleCIFAR10MCDropout,
)
