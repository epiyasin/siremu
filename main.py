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

# Set random seed for reproducibility
np.random.seed(settings["random_seed"])
torch.manual_seed(settings["random_seed"])

# Generate Agent-Based Model (ABM) data
all_realisations = generate_ABM_data(settings)

# Prepare data for Neural Network (NN)
X = []
y = []
for realisation in all_realisations:
    X.append(np.array([settings["scenario"], realisation["SIR_data"].values[:, 0]]))
    y.append(np.array(realisation["incidences"]))

# Split into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["test_pct"], random_state=settings["random_seed"])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=settings["val_pct"], random_state=settings["random_seed"])

# Convert to PyTorch DataLoaders
train_data = TensorDataset(torch.from_numpy(np.array(X_train)), torch.from_numpy(np.array(y_train)))
train_loader = DataLoader(train_data, shuffle=True, batch_size=settings["nn_batch_size"], num_workers=settings["num_workers"])

val_data = TensorDataset(torch.from_numpy(np.array(X_val)), torch.from_numpy(np.array(y_val)))
val_loader = DataLoader(val_data, shuffle=True, batch_size=settings["nn_batch_size"], num_workers=settings["num_workers"])

test_data = TensorDataset(torch.from_numpy(np.array(X_test)), torch.from_numpy(np.array(y_test)))
test_loader = DataLoader(test_data, shuffle=True, batch_size=settings["nn_batch_size"], num_workers=settings["num_workers"])

# Define model, criterion and optimizer
if settings["model_type"] == "FFNN":
    model = FFNN(settings["input_size"], settings["hidden_size"], settings["output_size"])
elif settings["model_type"] == "GRU":
    model = GRU(settings["input_size"], settings["hidden_size"], settings["output_size"])
else:
    raise ValueError(f"Invalid model_type: {settings['model_type']}. Choose from 'FFNN' or 'GRU'.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"])

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, settings["nn_epochs"])

# Run the emulator
predictions, actual = run_emulator(model, test_loader)

# Comparison mode: Run a single realisation of the ABM and compare it to the emulator's output
if settings["mode"] == "comparison":
    realisation = generate_ABM_data(settings)[0]
    X = torch.from_numpy(np.array([settings["scenario"], realisation["SIR_data"].values[:, 0]]))
    y_actual = np.array(realisation["incidences"])
    y_predicted = model(X).detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(y_actual, label='ABM')
    plt.plot(y_predicted, label='Emulator')
    plt.legend()
    plt.show()
