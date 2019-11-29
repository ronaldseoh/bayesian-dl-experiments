from torchvision.models.squeezenet import SqueezeNet as SqueezeNetTorchVision
from torchvision.models.squeezenet import model_urls
from torchvision.models.utils import load_state_dict_from_url


class SqueezeNet(SqueezeNetTorchVision):
    def __init__(self, version='1_1'):
        super(SqueezeNet, self).__init__(version)
        self.version = version

    def load_pretrained(self):
        arch = 'squeezenet' + self.version

        state_dict = load_state_dict_from_url(
            model_urls[arch],
            progress=True)

        self.load_state_dict(state_dict)
