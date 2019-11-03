import os
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.utils import download_url

class UCIDatasets(Dataset):

    uci_dataset_configs = {
        'yacht': {
            'url': "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            'features': np.arange(6),
            'targets': [6],
        },
         'bostonHousing': {
            'url': "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            'features': np.arange(13),
            'targets': [13],
        },
         'power-plant': {
            'url': "https://github.com/ronaldseoh/DropoutUncertaintyExps/raw/master/UCI_Datasets/power-plant/data/data.txt",
            'features': np.arange(4),
            'targets': [4],
        },
         'naval-propulsion-plant': {
            'url': "https://github.com/ronaldseoh/DropoutUncertaintyExps/raw/master/UCI_Datasets/naval-propulsion-plant/data/data.txt",
            'features': np.arange(16),
            'targets': [16],
        },
    }

    def __init__(self, dataset_name, root_dir, transform=None, download=True):

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform

        # Get the url and file name for the requested dataset_name
        self.url = self.uci_dataset_configs[self.dataset_name]['url']
        self.filename = os.path.basename(urlparse(self.url).path)

        if download:
            os.makedirs(os.path.join(self.root_dir, self.dataset_name), exist_ok=True)
            self.download()

        # Process the downloaded data
        fp = os.path.join(self.root_dir, self.dataset_name, self.filename)
        self.data = torch.Tensor(np.loadtxt(fp))

        # Store feature / target columns
        self.features = self.uci_dataset_configs[self.dataset_name]['features']
        self.targets = self.uci_dataset_configs[self.dataset_name]['targets']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data[idx][self.features], self.data[idx][self.targets])

        return sample

    def download(self):
        download_url(self.url, self.root_dir, os.path.join(self.dataset_name, self.filename))
