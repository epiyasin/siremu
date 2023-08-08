import os
import numpy as np
import torch
from ABM import generate_ABM_data
from config import settings

def generate_data():
    print("Generating data...")

    # Data generation
    np.random.seed(settings["execution"]["random_seed"])
    ABM_data = generate_ABM_data(settings)

    # Extract X, Y, and the rates from the ABM data
    X = []
    Y = []

    for realisation in ABM_data:
        X.append([
            np.random.uniform(*settings["ABM"]["infection_rate_range"]),
            np.random.uniform(*settings["ABM"]["recovery_rate_range"]),
            settings["ABM"]["population_size"]
        ])
        Y.append(realisation['incidences'])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

    data_to_save = {
        'X': X,
        'Y': Y,
        'ABM_data': ABM_data
    }
    torch.save(data_to_save, os.path.join(settings["data"]["data_dir"], 'ABM_data.pth'))
    return X, Y, ABM_data

def load_data():
    data_folder_path = settings["data"]["data_dir"]
    data_file_path = os.path.join(data_folder_path, 'ABM_data.pth')
    loaded_data = torch.load(data_file_path)
    X = loaded_data['X'].clone().detach()
    Y = loaded_data['Y'].clone().detach()
    ABM_data = loaded_data['ABM_data']

    return X, Y, ABM_data
