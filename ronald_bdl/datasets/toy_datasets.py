import torch
from torch.utils.data import Dataset

class ToyDatasets(Dataset):

    def __init__(self, random_seed=691, n_samples=50,
                 low=-5, high=5, mean=0, std=9, transform=None):

        self.transform = transform

        self._generator = torch.Generator()
        self._generator.manual_seed(random_seed)

        # Based on
        # https://github.com/pawni/BayesByHypernet/blob/master/toy_data.ipynb
        # https://pytorch.org/docs/master/tensors.html#torch.Tensor.uniform_
        # https://pytorch.org/docs/master/tensors.html#torch.Tensor.normal_
        self.data_x = torch.empty(n_samples,).uniform_(low, high, generator=self._generator)
        self.data_y = torch.pow(self.data_x, 3) + torch.empty(n_samples,).normal_(
            mean=mean, std=std, generator=self._generator)

        self.n_features = self.data_x.shape[1]
        self.n_targets = self.data_y.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) != list:
            idx_list = [idx]
        else:
            idx_list = idx
    
        sample = (self.data_x[idx_list], self.data_y[idx_list])

        return sample
