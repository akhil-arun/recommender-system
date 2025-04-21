import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, sparse_input: dict, dense_input: torch.Tensor, labels: torch.Tensor):
        self.sparse_input = sparse_input
        self.dense_input = dense_input
        self.labels = labels
        self.keys = list(sparse_input.keys())
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sparse = {key: self.sparse_input[key][idx] for key in self.keys}
        dense = self.dense_input[idx]
        label = self.labels[idx]
        return sparse, dense, label