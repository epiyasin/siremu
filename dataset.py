"""
Abstraction over emulator input data for use in NN training.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MintDataset(Dataset):
    def __init__(self, input_file):
        self.frame = pd.read_csv(input_file)
        self.nParams = 20
        self.outDims = 61

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        input_vals = torch.tensor(np.array(row[0:self.nParams]), dtype=torch.float32)
        output_vals = torch.tensor(np.array(row[self.nParams:]), dtype=torch.float32)
        return input_vals, output_vals
