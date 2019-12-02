from .fcnet import FCNet
from .fcnet_pyro import FCNetPyro
from .squeezenet_original import SqueezeNet
from .squeezenet_dropout import SqueezeNetDropout
from .simple_cifar10 import SimpleCIFAR10

__all__ = (
    FCNet,
    FCNetPyro,
    SqueezeNet,
    SqueezeNetDropout,
    SimpleCIFAR10,
)
