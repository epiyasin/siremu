import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from ABM import generate_ABM_data
from models import FFNN, GRU
from training import train_model
from testing import run_emulator

# User settings
settings = {
    "num_workers": 0,
    "random_seed": 42,
    "infection_rate_range": (0.1, 0.2),
    "recovery_rate_range": (0.05, 0.15),
    "population_size": 5000,
    "num_time_steps": 256,
    "num_realisations": 100,
    "num_iterations": 10,
    "nn_epochs": 256,
    "nn_batch_size": 32,
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 256,
    "learning_rate": 0.001,
    "model_type": "FFNN",  # FFNN or "GRU"
    "test_pct": 0.1,
    "val_pct": 0.1,
    "mode": "comparison",  # emulation or "comparison"
    "scenario": [0.15, 0.10, 1000]  # a specific scenario with the infection rate, recovery rate, and population size
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Data generation
    np.random.seed(settings["random_seed"])
    ABM_data = generate_ABM_data(settings)

    X = []
    Y = []

    for realisation in ABM_data:
        X.append([np.random.uniform(*settings["infection_rate_range"]), np.random.uniform(*settings["recovery_rate_range"]), settings["population_size"]])
        Y.append(realisation['incidences'])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

    # Split data into train, validation and test
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=settings["test_pct"], random_state=settings["random_seed"])
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=settings["val_pct"]/(1-settings["test_pct"]), random_state=settings["random_seed"])

    scaler = StandardScaler()
    
    # Normalize your input data
    X_train_scaled = scaler.fit_transform(X_train.numpy())
    X_val_scaled = scaler.transform(X_val.numpy())
    X_test_scaled = scaler.transform(X_test.numpy())

    # Convert normalized data back to tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Convert to tensor datasets
    train_data = TensorDataset(X_train, Y_train)
    val_data = TensorDataset(X_val, Y_val)
    test_data = TensorDataset(X_test, Y_test)

    
    # Convert to tensor datasets
    train_data = TensorDataset(X_train, Y_train)
    val_data = TensorDataset(X_val, Y_val)
    test_data = TensorDataset(X_test, Y_test)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])
    val_loader = DataLoader(val_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])
    test_loader = DataLoader(test_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])

    # Model selection
    if settings["model_type"] == 'FFNN':
        model = FFNN(settings["input_size"], settings["hidden_size"], settings["output_size"])
    elif settings["model_type"] == 'GRU':
        model = GRU(settings["input_size"], settings["hidden_size"], settings["output_size"])

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"])

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, settings["nn_epochs"])

    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')

    # Load the model when you want to run the emulator
    if settings["model_type"] == 'FFNN':
        model = FFNN(settings["input_size"], settings["hidden_size"], settings["output_size"])
    elif settings["model_type"] == 'GRU':
        model = GRU(settings["input_size"], settings["hidden_size"], settings["output_size"])

    model.load_state_dict(torch.load('model.pth'))

    if settings["mode"] == "comparison":
        # Run the emulator
        predictions, actual = run_emulator(model, test_loader)

        # Convert predictions and actual to numpy for easier handling
        predictions_np = np.concatenate(predictions, axis=0)
        actual_np = np.concatenate(actual, axis=0)

        # Select 25 random epidemics for plotting
        num_plots = 25
        indices = np.random.choice(range(predictions_np.shape[0]), size=num_plots, replace=False)

        # Create a 5x5 grid of subplots
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))

        for i, ax in enumerate(axes.flat):
            idx = indices[i]
            ax.plot(predictions_np[idx], label='Predicted')
            ax.plot(actual_np[idx], label='Actual')
            ax.set_title(f'Epidemic {idx+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('New Infections')
            ax.legend()

        # Adjust the layout so the plots do not overlap
        plt.tight_layout()
        plt.show()

    elif settings["mode"] == "emulation":
        # Define the specific scenario to emulate
        scenario = torch.tensor([settings["scenario"]], dtype=torch.float32)

        # Use the model to emulate the scenario
        with torch.no_grad():
            predicted_incidence = model(scenario)

        # Plot the predicted incidence
        plt.figure(figsize=(10, 5))
        plt.plot(predicted_incidence[0].numpy(), label='Predicted Incidence')
        plt.title('Emulated Epidemic')
        plt.xlabel('Time Step')
        plt.ylabel('New Infections')
        plt.legend()
        plt.show()