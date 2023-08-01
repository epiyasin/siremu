import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ABM import generate_ABM_data
from models import FFNN, GRU
from training import train_model
from testing import run_emulator

# User settings
settings = {
    "num_workers": 0,
    "random_seed": 42,
    "infection_rate_range": (0.10, 0.20),
    "recovery_rate_range": (0.05, 0.15),
    "population_size": 5000,
    "num_time_steps": 256,
    "num_realisations": 1000,
    "nn_epochs": 256,
    "nn_batch_size": 32,
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 256,
    "learning_rate": 0.001,
    "model_type": "GRU",  # FFNN or "GRU"
    "test_pct": 0.1,
    "val_pct": 0.1,
    "mode": "comparison",  # emulation or "comparison"
    "scenario": [0.15, 0.10, 1000]  # a specific scenario with the infection rate, recovery rate, and population size
}

# rest of main.py would be same as in previous script, making appropriate calls to functions from imported modules
