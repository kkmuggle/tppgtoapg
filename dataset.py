import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class PPGECGDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pickle.load(open(file_path, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        segment_data = self.data[idx]
        input_data = segment_data[:, [0, 1, 2]]
        output_data = segment_data[:, 4]  # Select ABP (4th index) signal

        input_data_transposed = np.transpose(input_data)
        output_data_expanded = np.expand_dims(output_data, axis=0)

        return torch.tensor(input_data_transposed, dtype=torch.float32), torch.tensor(output_data_expanded, dtype=torch.float32)