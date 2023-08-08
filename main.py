import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ABM import generate_ABM_data
from models import FFNN, GRU, LSTM, BiRNN
from training import train_model
from testing import run_emulator
from plotting import plot_comparison, plot_emulation
from utils import select_model, check_model_exists, check_data_exists, check_data_folder_exists
# User settings
settings = {
    "data": {
        "generate_ABM": False, # If True, generates Agent-Based Model (ABM) data; if False, uses data from a saved file (WARNING: If set to "True" and no data is detected it will run simulation program)
        "data_dir": os.path.join(os.getcwd(), "data"), # Directory containing the preprocessed ABM dataset (ideally siremu)
        "num_workers": 0, # Number of workers to use for loading data in DataLoader
        "shuffle": True, # If True, shuffles the data in DataLoader
        "test_pct": 0.2, # Fraction of data used for testing
        "val_pct": 0.2  # Fraction of data used for validation
    },
    "execution": {
        "max_workers": 16, # Maximum number of workers for ProcessPoolExecutor (optimal for current system configuration)
        "random_seed": 42, # Seed for random number generator to ensure reproducibility
        "mode": "comparison",  # Mode of operation: 'emulation' to emulate the ABM or 'comparison' to compare with other methods
        "cached_model": True # Used saved trained model (WARNING: If set to "True" and no cached model is detected it will run training program)
    },
    "ABM": {
        "infection_rate_range": (0.1, 0.5), # Range of daily infection rates to sample from
        "recovery_rate_range": (0.1, 0.5), # Range of daily recovery rates to sample from
        "population_size": 10000, # Total population size
        "num_time_steps": 256, # Number of time-series steps in ABM
        "num_realisations": 128, # Number of different realisations (i.e., simulations) for a given set of rates
        "num_iterations": 16, # Number of iterations to re-run the ABM with a fixed set of rates
        "scenario": [0.2, 0.3, 10000]  # A specific scenario detailing daily infection rate, daily recovery rate, and population size
    },
    "neural_net": {
        "nn_epochs": 256, # Number of training epochs
        "nn_batch_size": 64, # Number of samples per batch to load
        "input_size": 3, # Number of input neurons
        "hidden_size": 64, # Number of hidden neurons in the layer
        "output_size": 256, # Number of output neurons
        "model_type": "FFNN", # Type of neural network model: FFNN, GRU, LSTM or BiRNN
        "lr_scheduler": { 
            "learning_rate": 0.0001, # Initial learning rate for the optimizer
            "step_size": 64, # Number of epochs before changing the learning rate
            "gamma": 0.8 # Factor to reduce the learning rate by
        },
        "dropout_prob": 0.5
    },
    "plotting": {
        "num_plots": 9,  # Number of random epidemics for plotting in comparison mode
        "figure_size_comparison": (20, 20),  # Size of figure in comparison mode
        "figure_size_emulation": (10, 5),  # Size of figure in emulation mode
    }
}

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_folder_path = settings["data"]["data_dir"]

    # Check if the data folder exists and create it if not
    folder_exists = check_data_folder_exists(data_folder_path)

    # If the folder didn't exist, we know we need to generate data.
    # Otherwise, check if the data file exists within the folder
    should_generate_data = not folder_exists or (folder_exists and not check_data_exists(data_folder_path))

    if should_generate_data or settings["data"]["generate_ABM"]:
        print("Generating data...")

        # Data generation
        np.random.seed(settings["execution"]["random_seed"])
        ABM_data = generate_ABM_data(settings)

        # Extract X, Y, and the rates from the ABM data
        X = []
        Y = []
        infection_rates = []
        recovery_rates = []

        for realisation in ABM_data:
            X.append([
                np.random.uniform(*settings["ABM"]["infection_rate_range"]),
                np.random.uniform(*settings["ABM"]["recovery_rate_range"]),
                settings["ABM"]["population_size"]
            ])
            Y.append(realisation['incidences'])
            infection_rates.append(realisation['infection_rate'])
            recovery_rates.append(realisation['recovery_rate'])

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

        data_to_save = {
            'X': X,
            'Y': Y,
            'ABM_data': ABM_data
        }
        torch.save(data_to_save, os.path.join(settings["data"]["data_dir"], 'ABM_data.pth'))
            
    else:
        # Load the data from file
        data_file_path = os.path.join(data_folder_path, 'ABM_data.pth')
        loaded_data = torch.load(data_file_path)
        X = loaded_data['X']
        Y = loaded_data['Y']
        ABM_data = loaded_data['ABM_data']

        X = loaded_data['X'].clone().detach()
        Y = loaded_data['Y'].clone().detach()
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
    model = select_model(settings)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["neural_net"]["lr_scheduler"]["learning_rate"])

    model_exists = check_model_exists(settings["neural_net"]["model_type"])

    if model_exists and settings["execution"]["cached_model"]:
        # Load the saved model when you want to run the emulator
        model = select_model(settings)
        model.load_state_dict(torch.load(os.path.join("cached_models", settings["neural_net"]["model_type"] + 'model.pth')))
    else:
        if not model_exists:
            print("No saved models present, running training...")
            if not os.path.exists("cached_models"):
                os.makedirs("cached_models")

        # Train the model
        model = select_model(settings)  # Initialize the model for training
        train_model(model, criterion, optimizer, train_loader, val_loader, settings)
        # Save the model after training
        torch.save(model.state_dict(), os.path.join("cached_models", settings["neural_net"]["model_type"] + 'model.pth'))

    if settings["execution"]["mode"] == "comparison":
        # Run the emulator
        predictions, actual = run_emulator(model, test_loader)
        
        # Call the plotting function
        plot_comparison(predictions, actual, ABM_data, settings)

    elif settings["execution"]["mode"] == "emulation":
        # Emulate using the specific scenario
        emulation_input = torch.tensor([settings["ABM"]["scenario"]], dtype=torch.float32)
        
        # Normalize input
        emulation_input_scaled = scaler.transform(emulation_input.numpy())
        emulation_input = torch.tensor(emulation_input_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            emulation_output = model(emulation_input)

        # Call the plotting function for emulation
        plot_emulation(emulation_output, settings)