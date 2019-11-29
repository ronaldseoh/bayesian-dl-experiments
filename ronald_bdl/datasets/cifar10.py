import torchvision
import numpy as np


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self, root, transform=None, target_transform=None, download=False, limit_size=None):

        # Initialize torchvision.datasets.CIFAR10
        # Note: train=True to treat the original training set as the whole dataset
        super().__init__(
            root=root, train=True, 
            transform=transform, target_transform=target_transform, 
            download=download)
        
        # Randomly re-sample the dataset if limit_size is given
        if limit_size is not None:
            size = int(limit_size * len(self.data))

            random_indexes = np.random.randint(low=0, high=len(self.data), size=size)
            
            self.data = self.data[random_indexes, :, :, :]
            self.targets = [self.targets[i] for i in random_indexes]
