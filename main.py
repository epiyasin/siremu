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
from config import settings

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