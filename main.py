import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from ABM import generate_ABM_data
from models import FFNN, GRU, LSTM
from training import train_model
from testing import run_emulator

# User settings
settings = {
    "data": {
        "generate_ABM": True, # If True, generates Agent-Based Model (ABM) data; if False, uses data from a saved file
        "data_dir": "D:/Malaria/siremu/data", # Directory containing the preprocessed ABM dataset
        "num_workers": 0, # Number of workers to use for loading data in DataLoader
        "shuffle": True, # If True, shuffles the data in DataLoader
        "test_pct": 0.1, # Fraction of data used for testing
        "val_pct": 0.1  # Fraction of data used for validation
    },
    "execution": {
        "max_workers": 12, # Maximum number of workers for ProcessPoolExecutor (optimal for current system configuration)
        "random_seed": 42, # Seed for random number generator to ensure reproducibility
        "mode": "comparison"  # Mode of operation: 'emulation' to emulate the ABM or 'comparison' to compare with other methods
    },
    "ABM": {
        "infection_rate_range": (0.25, 2.5), # Range of daily infection rates to sample from
        "recovery_rate_range": (0.05, 0.5), # Range of daily recovery rates to sample from
        "population_size": 10000, # Total population size
        "num_time_steps": 256, # Number of time-series steps in ABM
        "num_realisations": 50, # Number of different realisations (i.e., simulations) for a given set of rates
        "num_iterations": 10, # Number of iterations to re-run the ABM with a fixed set of rates
        "scenario": [1.15, 0.10, 5000]  # A specific scenario detailing daily infection rate, daily recovery rate, and population size
    },
    "neural_net": {
        "nn_epochs": 256, # Number of training epochs
        "nn_batch_size": 32, # Number of samples per batch to load
        "input_size": 3, # Number of input neurons
        "hidden_size": 64, # Number of hidden neurons in the layer
        "output_size": 256, # Number of output neurons
        "model_type": "GRU", # Type of neural network model: FFNN, GRU, or LSTM
        "lr_scheduler": { 
            "learning_rate": 0.001, # Initial learning rate for the optimizer
            "step_size": 64, # Number of epochs before changing the learning rate
            "gamma": 0.8 # Factor to reduce the learning rate by
        }
    }
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if settings["data"]["generate_ABM"]:
        # Data generation
        np.random.seed(settings["execution"]["random_seed"])
        ABM_data = generate_ABM_data(settings)

        X = []
        Y = []

        for realisation in ABM_data:
            X.append([np.random.uniform(*settings["ABM"]["infection_rate_range"]), np.random.uniform(*settings["ABM"]["recovery_rate_range"]), settings["ABM"]["population_size"]])
            Y.append(realisation['incidences'])

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

        # Save the generated data
        np.savez(settings["data"]["data_dir"] + '/ABM_data.npz', X=X.numpy(), Y=Y.numpy())
    else:
        # Load the data from file
        loaded_data = np.load(settings["data"]["data_dir"] + '/ABM_data.npz')

        X = torch.tensor(loaded_data['X'], dtype=torch.float32)
        Y = torch.tensor(loaded_data['Y'], dtype=torch.float32)


    # Split data into train, validation and test
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=settings["data"]["test_pct"], random_state=settings["execution"]["random_seed"])
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=settings["data"]["val_pct"]/(1-settings["data"]["test_pct"]), random_state=settings["execution"]["random_seed"])

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

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=settings["neural_net"]["nn_batch_size"], shuffle=settings["data"]["shuffle"], num_workers=settings["data"]["num_workers"])
    val_loader = DataLoader(val_data, batch_size=settings["neural_net"]["nn_batch_size"], shuffle=settings["data"]["shuffle"], num_workers=settings["data"]["num_workers"])
    test_loader = DataLoader(test_data, batch_size=settings["neural_net"]["nn_batch_size"], shuffle=settings["data"]["shuffle"], num_workers=settings["data"]["num_workers"])

    # Model selection
    if settings["neural_net"]["model_type"] == 'FFNN':
        model = FFNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])
    elif settings["neural_net"]["model_type"] == 'GRU':
        model = GRU(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])
    elif settings["neural_net"]["model_type"] == 'LSTM':
        model = LSTM(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["neural_net"]["lr_scheduler"]["learning_rate"])

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, settings)
    
    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')

    # Load the model when you want to run the emulator
    if settings["neural_net"]["model_type"] == 'FFNN':
        model = FFNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])
    elif settings["neural_net"]["model_type"] == 'GRU':
        model = GRU(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])
    elif settings["neural_net"]["model_type"] == 'LSTM':
        model = LSTM(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"])

    model.load_state_dict(torch.load('model.pth'))

    if settings["execution"]["mode"] == "comparison":
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

    elif settings["execution"]["mode"] == "emulation":
        # Define the specific scenario to emulate
        scenario = torch.tensor([settings["ABM"]["scenario"]], dtype=torch.float32)

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