import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from settings import Settings

settings = Settings()

class MintDataset(Dataset):
    def __init__(self, input_file):
        self.frame = pd.read_csv(input_file)
        self.nParams = settings.neural_net.input_size
        self.outDims = settings.neural_net.output_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        input_vals = torch.tensor(np.array(row[0:self.nParams]), dtype=torch.float32)
        output_vals = torch.tensor(np.array(row[self.nParams:]), dtype=torch.float32)
        return input_vals, output_vals
