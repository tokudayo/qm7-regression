import scipy.io
from torch.utils.data import Dataset
import torch
import numpy as np


class QM7(Dataset):
    def __init__(self, path, train=True, split=-1):
        assert isinstance(split, int) and -1 <= split <= 4, 'Fold must be in range [-1, 4]'
        super().__init__()
        self.path = path
        data = scipy.io.loadmat(path)
        if split != -1:
            split = data['P'][split]
            mask = torch.zeros(data['T'].size, dtype=bool)
            mask[split] = True
            if train:
                mask = ~mask
            print(mask.shape)
            self.X = data['X'][mask]
            self.y = data['T'].T[mask]
            self.R = data['R'][mask]
        else:
            self.X = data['X']
            self.y = data.T['T']
            self.R = data['R']
        X_proc = []
        for x in self.X:
            X_proc.append(x[np.triu_indices(23)])
        self.X = np.vstack(X_proc)
        
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)