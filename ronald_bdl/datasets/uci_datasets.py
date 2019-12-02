import os
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import Dataset
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
         'protein-tertiary-structure': {
            'url': "https://github.com/ronaldseoh/DropoutUncertaintyExps/raw/master/UCI_Datasets/protein-tertiary-structure/data/data.txt",
            'features': np.arange(9),
            'targets': [9],
        },
    }

    def __init__(
        self, dataset_name, root_dir, limit_size=None,
            transform=None, target_transform=None, download=True):

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        # Get the url and file name for the requested dataset_name
        self.url = self.uci_dataset_configs[self.dataset_name]['url']
        self.filename = os.path.basename(urlparse(self.url).path)

        if download:
            os.makedirs(os.path.join(self.root_dir, self.dataset_name), exist_ok=True)
            self.download()

        # Process the downloaded data
        fp = os.path.join(self.root_dir, self.dataset_name, self.filename)

        self.data = torch.Tensor(np.loadtxt(fp))

        # Randomly re-sample the dataset if limit_size is given
        if limit_size is not None:
            size = int(limit_size * len(self.data))

            random_indexes = np.random.randint(low=0, high=len(self.data), size=size)
            self.data = self.data[random_indexes]

        # Store feature / target columns
        self.features = self.uci_dataset_configs[self.dataset_name]['features']
        self.n_features = len(self.features)
        self.targets = self.uci_dataset_configs[self.dataset_name]['targets']
        self.n_targets = len(self.targets)

        # Transform
        if self.transform is not None:
            self.X_mean = torch.mean(self.data[:, self.features], 0)
            self.X_std = torch.std(self.data[:, self.features], 0)

            self.y_mean = torch.mean(self.data[:, self.targets])
            self.y_std = torch.std(self.data[:, self.targets])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) != list:
            idx_list = [idx]
        else:
            idx_list = idx

        if self.transform is not None:
            X = self.transform(
                self.data[idx_list][:, self.features],
                mean=self.X_mean, std=self.X_std)
        else:
            X = self.data[idx_list][:, self.features]

        if self.target_transform is not None:
            y = self.target_transform(
                self.data[idx_list][:, self.targets],
                mean=self.y_mean, std=self.y_std)
        else:
            y = self.data[idx_list][:, self.targets]

        sample = (X, y)

        return sample

    def download(self):
        download_url(
            self.url, self.root_dir,
            os.path.join(self.dataset_name, self.filename))
